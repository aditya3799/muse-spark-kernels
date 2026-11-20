with open("submission_s108.py") if False else open("submission_s122_att_qdir_pk2.py") as f:
    pass

import os
# Check what submission files exist
sub_files = [f for f in os.listdir(".") if f.startswith("submission") and f.endswith(".py")]
print("Available submission files:", sub_files)
