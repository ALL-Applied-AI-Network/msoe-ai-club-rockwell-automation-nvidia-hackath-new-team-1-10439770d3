"""Drop-in wrapper for the validated adaptive one-wall bin-lift policy.

Copy this file to the organizer client's ``policy.py`` together with the
``robot_control`` dependencies.  The policy requests table/side/wrist RGB,
table/wrist depth, and stops at a verified 10 mm lift-and-hold target.
"""

from robot_control.adaptive_bin_policy import AdaptiveBinLiftPolicy as Policy
