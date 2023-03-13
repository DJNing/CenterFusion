CF_ROOT=/home/jianning/RC_fusion/CenterFusion
# git clone --recursive https://github.com/DJNing/CenterFusion.git $CF_ROOT
cd $CF_ROOT


pip install -r requirements.txt

# cd $CF_ROOT/src/lib/model/networks
# git clone https://github.com/jinfagang/DCNv2_latest.git
cd $CF_ROOT/src/lib/model/networks/DCNv2_latest
python3 setup.py build develop
