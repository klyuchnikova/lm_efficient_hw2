# lm_efficient_hw2

Удачи с проверкой -_-
Конду я не пробовала подключать, в теории знаю что 

conda create -n triton_env python=3.9
conda activate triton_env
conda install -c conda-forge numpy pytorch transformers tritonclient requests
conda clean -a -y
conda pack -n triton_env -o environment.tar.gz -f
conda deactivate

и запускать тритон в композере --env PYTHONNOUSERSITE=TRUE \
  --env PYTHONPATH=/path/to/conda-environment/lib/python3.9/site-packages \
  --env LD_LIBRARY_PATH=/path/to/conda-environment/lib

я успела проверить инференс с тритоном только onnx 32. Так я тестирование проводила все на каггле так как мой комп чуть не сдох
(и правда fp16 быстрее в пару раз когда результаты те же)