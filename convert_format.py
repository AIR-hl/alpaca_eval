import argparse
import os

from datasets import load_dataset


def convert_format(data_dir, output_path, model_name):
    # 加载数据集
    dataset = load_dataset(data_dir, split='train').select_columns(['messages'])

    def func(row):
        row['instruction']= row['messages'][0]['content']
        row['output'] = row['messages'][1]['content']
        row['generator'] = model_name
        return row
    dataset = dataset.map(func)
    dataset = dataset.remove_columns(['messages'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dataset.to_json(f"{output_path}/outputs.json", orient='records', lines=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset format")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model")

    args = parser.parse_args()

    convert_format(args.data_dir, args.output_path, args.model_name)
