//if you add package, you should to run bellow command
pip3 freeze > package.txt

pip3 install -r ./package.txt –user


1. FeatureExtract/EQNormalizer.py
2. FeatureExtract/ExcelAugmantation.py -> 회사 git 삭제 확인필요
3. FeatureExtract/FeatureExtractMacro.py 
-> 자체 실행 
	->엑셀 + 음원 파일 -> 특징추출정리(data.npy) comp
3. FeatureExtract/FeatureExtractMacroEQ.py 
-> 자체 실행 
	->엑셀 + 음원 파일 -> 특징추출정리(dataEQ.npy) EQ
4. FeatureExtract/FeatureGetter.py
-> 음원 특징 추출 (call FeatureCollecter)
5. FeatureExtract/Normalizer.py

6. LearningModule/DataLoader.py , LearningModule/Learner.py X

7. LearningModule/ModelWrapper.py
Comp model Trainning

8. LearnigModule/ModelWrapperEQ.py
EQ Model Trainning

9. LearningModule/TuningData.py
normalize 안씀
TuningModule에 넘겨주는 comp data set
10. Test/DataIntegrityChecker.py

11. Test/ modelTest.py

12. Test/ testModule.py

13. Test/ VideoTranster.py


Service Logic
AudioTuneDeamon -> MasteringRPC.py

Train Logic 
LearningModule/ModelWrapper.py
LearnigModule/ModelWrapperEQ.py

Data Generation Logic
FeatureExtract/FeatureExtractMacro.py 
FeatureExtract/FeatureExtractMacroEQ.py 



1, frontend
npm run serve

2. backend
npm start

3. python
source ~/venv/bin/activate
python3 AudioTuneDaemon.py

4. port  config

- port 3010 : connect point between frondend and backend
backend/bin/www -> var port = normalizePort(process.env.PORT || '3010');
<<<<<<< HEAD
frontend/src/components -> url: 'http://221.146.63.149:3010/api/upload',
=======
frontend/src/components/DragAndDropZone.vue -> url: 'http://221.146.63.54:3010/api/upload',
>>>>>>> 981f811596211a461807adce49024a627647a47a

- port 4242 : connect point between backend`s nodejs and python`s zerorpc
backend/routes/fileReceiver.js -> client.connect("tcp://127.0.0.1:4242");
python_module/AudioTuneDaemon.py -> s.bind("tcp://0.0.0.0:4242")

- ip address
<<<<<<< HEAD
frontend/vue.config.js -> public : '221.146.63.149',
=======
frontend/vue.config.js -> public : '221.146.63.54',

5. firewall
sudo ufw allow ${port_num}
>>>>>>> 981f811596211a461807adce49024a627647a47a
