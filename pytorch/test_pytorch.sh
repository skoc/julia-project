# git clone https://github.com/HiLab-git/DTC.git 
cp requirements.txt DTC
cp discriminator.py DTC/code/networks
cd DTC
# python3 -m pip install -r requirements.txt
cd code
# python3 train_la_dtc.py
python3 test_LA.py