Replece above files in original monodepth [1] for midair dataset finetuning

[1] https://github.com/nianticlabs/monodepth2


Commands
----
```bash
--- train ---
python train.py --model_name finetuned_mono --dataset midair --data_path MidAir --split midair

--- test single image---
python test_simple.py --image_path ./test/test.jpg --model_name mono+stereo_640x192
python test_simple.py --image_path ./data/b1_test --model_name weights_19 --ext png
```

args
----
checkpoints path: "/tmp/"

--dataset: A class defined in "./dataset/monodataset.py"

--dataset: Dataset folder name "./xxxx"

--split txt file path with dataset image info "./splits/xxxx/"

----

Use a new dataset
----
1. create a class in  "./dataset/monodataset.py" 
		`like this one "class MidAirDataset(MonoDataset)"
2. import your new class in  "./dataset/__init__.py"
3. create split file "train_file.txt" and "val_file.txt"
		`which contain {folder,name,l/r} for all images from dataset
		`then must remove 1st and last info if using mono mode
4. add relative choice to "./options.py"
