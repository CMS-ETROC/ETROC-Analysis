## Procedure
### Condor submission only available on LXPLUS
### Require python >= 3.9

#### Clone ETROC-Analysis repository
ssh to Lxplus, then move to afs work directory

```cd /afs/cern.ch/work/<first alphabet of the username>/<username>```

```mkdir -p ETROC && cd ETROC```

```git clone https://github.com/CMS-ETROC/ETROC-Analysis.git```

Then go to the directory where control submitting jobs.

```cd ETROC-Analysis/TestBeam/condor_at_lxplus```

Let's load python 3.9 enviornment if you're on the server. (e.g. Lxplus)

```source load_python39.sh```

Install libraries:

```python -m pip install --user crc natsort lmfit```

Examples commands are in run_scripts directory.