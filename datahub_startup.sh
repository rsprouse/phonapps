#!/usr/bin/env bash

# Temporary hack for startup if we are running on datahub.berkeley.edu.
# OAuth environment variables are not set on mybinder.org. Since
# datahub is not a binderhub we must install some packages and do some
# monkeypatching.

if [ ! -z "$OAUTH2_LOGIN_ID" ]
then
  cd $HOME
  logger="${HOME}/datahub_start.phonapps.log"
  pip install praat-parselmouth > $logger 2>&1
  pip install git+https://github.com/rsprouse/phonlab >> $logger 2>&1
  pip install git+https://github.com/rsprouse/bokeh_phon >> $logger 2>&1

  # Monkeypatch default_jupyter_url value.
  perl -pi -e 's#hub.gke.mybinder.org#datahub.berkeley.edu#' /opt/conda/lib/python3.8/site-packages/bokeh_phon/utils.py >> $logger 2>&1
  rm /opt/conda/lib/python3.8/site-packages/bokeh_phon/__pycache__/utils*.pyc >> $logger 2>&1
  echo Setup results can be found in $logger.
fi
