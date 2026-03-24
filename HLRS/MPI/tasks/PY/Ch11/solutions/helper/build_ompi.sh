#!/usr/bin/env bash
python3 = $(which python3)
CC=mpicc python3 split_helper_ompi_build.py
