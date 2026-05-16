nohup .venv/bin/python3 scripts/run_global.py configs/global_spinup.yaml \
    > output/spinup/run.log 2>&1 &
echo $!   # save this PID
