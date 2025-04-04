e:
	python run_simulation.py --strategy epipolar --sample-locations ../eval/sample_locations_valid.npy --source-poses ../eval/source_view_poses.npy --target-pose ../eval/target_view_pose.npy

s:
	python run_simulation.py --strategy spatial 

d:
	python debug_epipolar.py --sample-locations ../eval/sample_locations_valid.npy --source-poses ../eval/source_view_poses.npy --target-pose ../eval/target_view_pose.npy