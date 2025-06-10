#!/bin/bash

gunicorn -w 4 -b 0.0.0.0:8005 main:app