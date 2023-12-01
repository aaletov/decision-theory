#!/bin/bash
set -e
cd docs
xelatex translation
biber translation
xelatex translation
xelatex translation

xelatex report
biber report
xelatex report
xelatex report
