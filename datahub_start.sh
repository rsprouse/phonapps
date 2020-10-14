#!/usr/bin/env bash

# Temporary hack for startup if we are running on datahub.berkeley.edu.
# OAuth environment variables are not set on mybinder.org. Since
# datahub is not a binderhub we must install some packages to create
# the environment we need.

if [ ! -z "$OAUTH2_LOGIN_ID" ]
then
  cd $HOME
  logger="${HOME}/datahub_start.phonapps.log"
  date >> $logger 2>&1
  pip install pyarrow==0.15 >> $logger 2>&1
  pip install praat-parselmouth >> $logger 2>&1
  pip install git+https://github.com/rsprouse/audiolabel >> $logger 2>&1
  pip install git+https://github.com/rsprouse/phonlab >> $logger 2>&1
  pip install git+https://github.com/rsprouse/bokeh_phon >> $logger 2>&1

  echo Setup results can be found in $logger.
fi
