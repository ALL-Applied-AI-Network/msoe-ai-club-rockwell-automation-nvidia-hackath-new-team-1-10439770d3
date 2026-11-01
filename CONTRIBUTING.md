# Contributing

Issues and pull requests are welcome for reproducibility fixes, tests, documentation, and hardware-free improvements.

Before opening a pull request:

1. run `python scripts/check_public_release.py`;
2. run the tests in [`docs/reproduction.md`](docs/reproduction.md);
3. label evidence as physical, simulation, offline, or design;
4. do not commit credentials, endpoints, raw facility media, private telemetry, browser state, model checkpoints, or generated caches;
5. keep simulator ground truth out of policy inputs; and
6. explain any changed safety or motion assumption.

Physical robot changes require a separate site review. A merged pull request is not authorization to operate a robot.

