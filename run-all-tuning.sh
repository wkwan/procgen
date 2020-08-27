export OUTPUTS_DIR=./outputs
export RAY_MEMORY_LIMIT=60129542144
export RAY_CPUS=8
export RAY_STORE_MEMORY=30000000000

export EXPERIMENT="experiments/tune-ppo.yaml"
python3 train-ppo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-sac.yaml"
python3 train-sac.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

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

export EXPERIMENT="experiments/tune-impala.yaml"
python3 train-tune-impala.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-appo.yaml"
python3 train-tune-appo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-ddppo.yaml"
python3 train-tune-ddppo.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}

export EXPERIMENT="experiments/tune-marwil.yaml"
python3 train-tune-marwil.py -f ${EXPERIMENT} --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}
