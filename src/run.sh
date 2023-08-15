# python preprocess.py --data=../data/data_aaai.pkl  --dataset=avec

python train.py --model_name=model --data=../data/data_aaai.pkl --from_begin --device=cuda --epochs=200 --batch_size=64

python prediction.py --model_name=model --data=../data/data_aaai.pkl --device=cuda --epochs=200 --batch_size=64