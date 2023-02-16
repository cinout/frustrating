import os
from collections import defaultdict
import json
import tempfile


fileid = "mvtc_fdsa"

aa = fileid + (".png" if fileid.startswith("mvtec") else ".jpg")

print(aa)
