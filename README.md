# NetVLAD

[NetVLAD](https://arxiv.org/abs/1511.07247) implementation using pytorch
-------------------------------------

based on [Nanne/pytorch-Netvlad](https://github.com/Nanne/pytorch-NetVlad)

[NetVLAD 논문리뷰](https://jhyeup.tistory.com/entry/NetVLAD-CNN-architecture-for-weakly-supervised-place-recognition)

naver labs 챌린지에서 제공하는 데이터셋중 영상 데이터를 사용하였습니다.
차량에서 수집한 스테레오 영상과 각 영상이 촬영된 시점의 차량의 자세 정보를 이용합니다.

논문에서는 Google Street View Time Machine의 GPS데이터를 사용하지만 구현에 사용된 데이터셋에는 GPS대신 자세 정보를 사용합니다.
차량의 자세 정보(6-DOF)를 이용해 데이터가 처음 촬영된 시점을 (0,0,0)으로 하여 각각의 영상이 촬영된 가상의 좌표를 계산하여 거리에 따라 positive와 negative를 나누어 사용합니다.
