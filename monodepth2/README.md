Replece above files in original monodepth [1] for midair dataset finetuning

[1] https://github.com/nianticlabs/monodepth2


Commands
----
train
```bash
python train.py --model_name finetuned_mono --dataset midair --data_path MidAir --split midair
```
checkpoints path: "/tmp/"

test single image

```bash
python test_simple.py --image_path ./test/test.jpg --model_name mono+stereo_640x192
python test_simple.py --image_path ./data/b1_test --model_name weights_19 --ext png
```

Args
```
--dataset 		# A class defined in "./dataset/monodataset.py"
--data_path 		# Dataset folder name "./xxxx"
--split 		# txt file path with dataset image info "./splits/xxxx/"
```

Use a new dataset
----
1. Create a class in  "./dataset/monodataset.py" 
		`like this one "class MidAirDataset(MonoDataset)"
2. Import your new class in  "./dataset/__init__.py"
3. Create split file "train_file.txt" and "val_file.txt", which contain {folder,name,l/r} for all images from dataset.
4. Then must remove 1st and last info if using mono mode in step 3.
5. Add relative choice to "./options.py"
