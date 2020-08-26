export OUTPUTS_DIR=./outputs
export RAY_MEMORY_LIMIT=60129542144
export RAY_CPUS=8
export RAY_STORE_MEMORY=30000000000

export EXPERIMENT="experiments/tune-ppo.yaml"
python3 train-ppo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-sac.yaml"
python3 train-sac.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-ddpg.yaml"
# python3 train-ddpg.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-apex_ddpg.yaml"
# python3 train-apex_ddpg.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-td3.yaml"
# python3 train-tune-td3.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-es.yaml"
# python3 train-tune-es.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-ars.yaml"
# python3 train-tune-ars.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-dqn.yaml"
python3 train-tune-dqn.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-simpleq.yaml"
python3 train-simpleq.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-apex.yaml"
python3 train-tune-apex.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-a3c.yaml"
python3 train-tune-a3c.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-a2c.yaml"
python3 train-tune-a2c.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-pg.yaml"
# python3 train-tune-pg.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-impala.yaml"
python3 train-tune-impala.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-qmix.yaml"
# python3 train-tune-qmix.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-appo.yaml"
python3 train-tune-appo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-ddppo.yaml"
python3 train-tune-ddppo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-marwil.yaml"
python3 train-tune-marwil.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-maml.yaml"
# python3 train-tune-maml.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

# export EXPERIMENT="experiments/tune-mbmpo.yaml"
# python3 train-tune-mbmpo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-dreamer.yaml"
python3 train-tune-dreamer.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

