dataset_info = dict(
    dataset_name='boxcars',
    paper_info=dict(
        title='BoxCars: 3D Boxes as CNN Input for Improved Fine-Grained Vehicle Recognition',
        container='CVPR',
        year='2016'
    ),
    keypoint_info={
        # keypoints can be swapped by flipping images
        0:
        dict(name='front_lpr_left', id=0, color=[255, 0, 0], type='vehicle-front', swap='front_lpr_right'),
        1:
        dict(name='front_light_left', id=1, color=[255, 0, 0], type='vehicle-front', swap='front_light_right'),
        2:
        dict(name='A_pillar_left', id=2, color=[255, 0, 0], type='vehicle-front', swap='A_pillar_right'),
        3:
        dict(name='front_roof_left', id=3, color=[255, 0, 0], type='vehicle-mid', swap='front_roof_right'),
        4:
        dict(name='rear_roof_left', id=4, color=[255, 0, 0], type='vehicle-mid', swap='rear_roof_right'),
        5:
        dict(name='rear_light_left', id=5, color=[255, 0, 0], type='vehicle-rear', swap='rear_light_right'),
        6:
        dict(name='rear_wheel_left', id=6, color=[255, 0, 0], type='vehicle-rear', swap='rear_wheel_right'),
        7:
        dict(name='front_wheel_left', id=7, color=[255, 0, 0], type='vehicle-front', swap='front_wheel_right'),
        # -----
        8:
        dict(name='front_lpr_right', id=8, color=[255, 0, 0], type='vehicle-front', swap='front_lpr_left'),
        9:
        dict(name='front_light_right', id=9, color=[255, 0, 0], type='vehicle-front', swap='front_light_left'),
        10:
        dict(name='A_pillar_right', id=10, color=[255, 0, 0], type='vehicle-front', swap='A_pillar_left'),
        11:
        dict(name='front_roof_right', id=11, color=[255, 0, 0], type='vehicle-mid', swap='front_roof_left'),
        12:
        dict(name='rear_roof_right', id=12, color=[255, 0, 0], type='vehicle-mid', swap='rear_roof_left'),
        13:
        dict(name='rear_light_right', id=13, color=[255, 0, 0], type='vehicle-rear', swap='rear_light_left'),
        14:
        dict(name='rear_wheel_right', id=14, color=[255, 0, 0], type='vehicle-rear', swap='rear_wheel_left'),
        15:
        dict(name='front_wheel_right', id=15, color=[255, 0, 0], type='vehicle-front', swap='front_wheel_left'),
        # -----
        16:
        dict(name='rear_lpr_left', id=16, color=[255, 0, 0], type='vehicle-rear', swap='rear_lpr_right'),
        17:
        dict(name='rear_lpr_right', id=17, color=[255, 0, 0], type='vehicle-rear', swap='rear_lpr_left'),
    },
    skeleton_info={
        0:
        dict(link=('front_lpr_left', 'front_lpr_right'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('front_light_left', 'A_pillar_left'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('A_pillar_left', 'front_roof_left'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('front_roof_left', 'rear_roof_left'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('rear_roof_left', 'rear_light_left'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('rear_wheel_left', 'front_wheel_left'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('front_light_right', 'A_pillar_right'), id=6, color=[0, 255, 0]),
        7:
        dict(link=('A_pillar_right', 'front_roof_right'), id=7, color=[255, 128, 0]),
        8:
        dict(link=('front_roof_right', 'rear_roof_right'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('rear_roof_right', 'rear_light_right'), id=9, color=[51, 153, 255]),
        10:
        dict(link=('rear_wheel_right', 'front_wheel_right'), id=10, color=[51, 153, 255]),
        11:
        dict(link=('rear_lpr_left', 'rear_lpr_right'), id=11, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025
    ])
