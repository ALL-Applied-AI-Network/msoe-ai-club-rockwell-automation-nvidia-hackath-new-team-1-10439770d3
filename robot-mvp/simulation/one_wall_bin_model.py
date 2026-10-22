"""Geometry and acceptance criteria for the one-wall bin-grip experiment.

This module is intentionally independent of Isaac Sim so its assumptions can be
unit-tested on the host before an expensive simulator run.  Lengths are metres,
masses are kilograms, and poses are expressed in the FANUC robot-base frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinSpec:
    """Measured/CAD dimensions of the official hackathon bin."""

    outer_x: float = 0.050
    outer_y: float = 0.050
    height: float = 0.055
    pocket_x: float = 0.035
    pocket_y: float = 0.035
    pocket_depth: float = 0.025
    density_kg_m3: float = 1240.0  # representative PLA; sweep mass in Isaac

    @property
    def wall_x(self) -> float:
        return (self.outer_x - self.pocket_x) / 2.0

    @property
    def wall_y(self) -> float:
        return (self.outer_y - self.pocket_y) / 2.0

    @property
    def floor_height(self) -> float:
        return self.height - self.pocket_depth

    @property
    def floor_volume_m3(self) -> float:
        return self.outer_x * self.outer_y * self.floor_height

    @property
    def wall_volume_m3(self) -> float:
        outer_area = self.outer_x * self.outer_y
        inner_area = self.pocket_x * self.pocket_y
        return (outer_area - inner_area) * self.pocket_depth

    @property
    def volume_m3(self) -> float:
        return self.floor_volume_m3 + self.wall_volume_m3

    @property
    def estimated_mass_kg(self) -> float:
        return self.volume_m3 * self.density_kg_m3

    @property
    def center_of_mass_z(self) -> float:
        floor_z = self.floor_height / 2.0
        wall_z = self.floor_height + self.pocket_depth / 2.0
        return (
            self.floor_volume_m3 * floor_z + self.wall_volume_m3 * wall_z
        ) / self.volume_m3


@dataclass(frozen=True)
class GripPlan:
    """A vertical straddle grasp centred on the bin's south wall."""

    bin_x: float = 0.450
    bin_y: float = 0.000
    table_z_robot: float = -0.0051
    # The failed physical run reached almost 4 mm below its commanded z.  A
    # 14 mm nominal insertion retains 11 mm above the 25 mm-deep pocket floor,
    # or roughly 7 mm after the largest observed tracking overshoot.
    insertion_depth: float = 0.014
    approach_clearance: float = 0.060
    lift_height: float = 0.010

    def wall_center_xy(self, spec: BinSpec) -> tuple[float, float]:
        return (
            self.bin_x,
            self.bin_y - spec.outer_y / 2.0 + spec.wall_y / 2.0,
        )

    def grip_tip(self, spec: BinSpec) -> tuple[float, float, float]:
        x, y = self.wall_center_xy(spec)
        wall_top = self.table_z_robot + spec.height
        return x, y, wall_top - self.insertion_depth

    def approach_tip(self, spec: BinSpec) -> tuple[float, float, float]:
        x, y, z = self.grip_tip(spec)
        return x, y, z + self.approach_clearance

    def lifted_tip(self, spec: BinSpec) -> tuple[float, float, float]:
        x, y, z = self.grip_tip(spec)
        return x, y, z + self.lift_height

    def minimum_tip_clearance_over_floor(self, spec: BinSpec) -> float:
        tip_z = self.grip_tip(spec)[2]
        pocket_floor_z = self.table_z_robot + spec.floor_height
        return tip_z - pocket_floor_z


def assess_trial(
    *,
    commanded_lift_mm: float,
    measured_lift_mm: float,
    max_downward_slip_mm: float,
    max_tilt_deg: float,
    jaw_gap_mm: float,
    ik_failures: int = 0,
    minimum_lift_fraction: float = 0.80,
    maximum_slip_mm: float = 1.0,
    maximum_tilt_deg: float = 5.0,
) -> dict:
    """Apply the pre-declared pass/fail gate and return auditable checks."""

    checks = {
        "ik_and_execution": ik_failures == 0,
        "lift": measured_lift_mm >= commanded_lift_mm * minimum_lift_fraction,
        "slip": max_downward_slip_mm <= maximum_slip_mm,
        "tilt": max_tilt_deg <= maximum_tilt_deg,
        # The wall is nominally 7.5 mm thick. A nonzero contact gap in this
        # broad range catches both a missed wall and implausible jaw penetration.
        "contact_gap": 5.5 <= jaw_gap_mm <= 10.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_lift_fraction": minimum_lift_fraction,
            "maximum_slip_mm": maximum_slip_mm,
            "maximum_tilt_deg": maximum_tilt_deg,
            "contact_gap_mm": [5.5, 10.0],
        },
    }


def default_manifest() -> dict:
    spec = BinSpec()
    plan = GripPlan()
    return {
        "bin": {
            "outer_mm": [spec.outer_x * 1000, spec.outer_y * 1000, spec.height * 1000],
            "pocket_mm": [
                spec.pocket_x * 1000,
                spec.pocket_y * 1000,
                spec.pocket_depth * 1000,
            ],
            "wall_mm": [spec.wall_x * 1000, spec.wall_y * 1000],
            "solid_volume_cm3": spec.volume_m3 * 1e6,
            "estimated_pla_mass_g": spec.estimated_mass_kg * 1000,
            "estimated_com_z_mm": spec.center_of_mass_z * 1000,
        },
        "grip": {
            "wall": "south",
            "wall_center_robot_m": list(plan.wall_center_xy(spec)),
            "grip_tip_robot_m": list(plan.grip_tip(spec)),
            "approach_tip_robot_m": list(plan.approach_tip(spec)),
            "lifted_tip_robot_m": list(plan.lifted_tip(spec)),
            "tip_clearance_over_pocket_floor_mm": (
                plan.minimum_tip_clearance_over_floor(spec) * 1000
            ),
        },
    }
