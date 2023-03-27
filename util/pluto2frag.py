#!/usr/bin/env python

import sys

def writefrag(outname, prefix, outlines):
    print("Write '{}{}.jl'".format(prefix, outname))
    with open("{}{}.jl".format(prefix, outname), "w") as outf:
        for outl in outlines:
            outf.write(outl)

def main(fname, prefix):
    print("fname: {}".format(fname))
    print("prefix: {}".format(prefix))
    outlines = []
    outname = None
    with open(fname) as f:
        for line in f.readlines():
            if line.startswith("#x-") and outname is not None:
                writefrag(outname, prefix, outlines)
                outname = None
                outlines = []
            if outname is not None:
                outlines.append(line.replace("\t", "  "))
            if line.startswith("end") and outname is not None:
                writefrag(outname, prefix, outlines)
                outname = None
                outlines = []
            if line.startswith("#x: "):
                outname = line[4:].strip()

if __name__ == '__main__':
    main(*sys.argv[1:])
