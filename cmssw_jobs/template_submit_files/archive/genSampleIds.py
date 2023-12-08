#!/usr/bin/env python3
import sys

for sampleId in sys.argv[1:]:
    for subSampleId in range(10):
        print(str(sampleId) + ", " + str(subSampleId))
