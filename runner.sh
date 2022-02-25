#cd /home/mryf/PycharmProjects/dgcl
#source venv/bin/activate
python main.py --log-dir ./logs/our_diva_no_change/0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=0 &
python main.py --log-dir ./logs/our_diva_no_change/1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=1 &
python main.py --log-dir ./logs/our_diva_no_change/2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=2 &
python main.py --log-dir ./logs/our_diva_no_change/3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=3
python main.py --log-dir ./logs/our_diva_no_change/4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=4 &
python main.py --log-dir ./logs/our_diva_no_change/5 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=5 &
python main.py --log-dir ./logs/our_diva_no_change/6 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=6 &
python main.py --log-dir ./logs/our_diva_no_change/7 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=7
python main.py --log-dir ./logs/our_diva_no_change/8 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=8 &
python main.py --log-dir ./logs/our_diva_no_change/9 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override seed=9