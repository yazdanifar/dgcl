python main.py --log-dir ./logs/our_diva_bce/scale_1lr001s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.001|seed=0" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr001s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.001|seed=1" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr001s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.001|seed=2" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr001s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.001|seed=3"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr001s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.001|seed=4" &
sleep 10

python main.py --log-dir ./logs/our_diva_bce/scale_1lr005s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.005|seed=0" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr005s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.005|seed=1" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr005s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.005|seed=2"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr005s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.005|seed=3" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr005s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.005|seed=4" &
sleep 10

python main.py --log-dir ./logs/our_diva_bce/scale_1lr01s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.01|seed=0" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr01s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.01|seed=1"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr01s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.01|seed=2" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr01s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.01|seed=3" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_1lr01s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=420|model.aux_loss_multiplier_d=200|model.beta_d=0.1|model.beta_x=0.1|model.beta_y=0.1|model.lr=0.01|seed=4" &
sleep 10


python main.py --log-dir ./logs/our_diva_bce/scale_05lr001s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.001|seed=0"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr001s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.001|seed=1" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr001s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.001|seed=2" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr001s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.001|seed=3" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr001s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.001|seed=4"
sleep 10


python main.py --log-dir ./logs/our_diva_bce/scale_05lr002s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.002|seed=0" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr002s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.002|seed=1" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr002s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.002|seed=2" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr002s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.002|seed=3"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr002s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.002|seed=4" &
sleep 10


python main.py --log-dir ./logs/our_diva_bce/scale_05lr005s0 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.005|seed=0" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr005s1 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.005|seed=1" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr005s2 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.005|seed=2"
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr005s3 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.005|seed=3" &
sleep 10
python main.py --log-dir ./logs/our_diva_bce/scale_05lr005s4 --config configs/ourdiva.yaml --episode episodes/diva_experiment/diva_unsupervised_boost.yaml --override "model.aux_loss_multiplier_y=210|model.aux_loss_multiplier_d=100|model.beta_d=0.05|model.beta_x=0.05|model.beta_y=0.05|model.lr=0.005|seed=4"







