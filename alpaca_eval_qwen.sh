 #!/bin/bash

# 设置模型名称变量
MODEL_NAME="mistral-ipo-vllm1"

# 设置文件路径变量，使用模型名称
DATA_PATH="autodl-tmp/inference/alpaca_eval-$MODEL_NAME"
OUTPUTS_PATH="autodl-tmp/rewards/alpaca_eval-$MODEL_NAME"
ANNOTATORS_CONFIG="weighted_alpaca_eval_qwen_plus"

# 设置环境变量
export OPENAI_API_KEY='sk-0d5547bf848947c0aac22dbf821dfe8d'
export OPENAI_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'

# 运行 Python 脚本
python convert_format.py --data_dir "$DATA_PATH" --output_path "$OUTPUTS_PATH" --model_name "$MODEL_NAME"

# 运行 Alpaca Eval
alpaca_eval --model_outputs "$OUTPUTS_PATH/outputs.json" --annotators_config "$ANNOTATORS_CONFIG"


# 计算总花费
python3 -c "
import pandas as pd

# 读取数据集
ds = pd.read_json('$OUTPUTS_PATH/$ANNOTATORS_CONFIG/annotations.json', orient='records', lines=False)

# 计算总花费
total_cost = ds[[col for col in ds.columns if col.endswith('price_per_example')][0]].sum()

# 保存到txt文件
with open('$OUTPUTS_PATH/$ANNOTATORS_CONFIG/cost.txt', 'w') as f:
    f.write(f'Total cost: {total_cost}\n')
"