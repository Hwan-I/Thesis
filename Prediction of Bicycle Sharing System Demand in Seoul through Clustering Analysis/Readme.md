## This has codes for clustering method that thesis proposed. 

### Description

##### 사용방법(clustering 기준)
* 여기있는 파일을 그대로 전부 다운 받습니다.
* data 폴더는 cluster를 만드는데 사용되는 폴더로 압축을 풀고 그대로 쓰시면 됩니다.
* result는 cluster 결과를 저장하는 파일로 밑의 main에서 위치는 변경하실 수도 있습니다.

##### main.py 
* new_clustering.py 파일을 실행시키는 파일입니다.
* 변수설명은 다음과 같습니다.
  * nc.new_clustering 선언시
    * cluster_save_path : cluster 결과를 저장하는 폴더 위치를 지정합니다. step1, 2, 3 파일 모두 pickle 형식으로 생성됩니다.
    * w_str : 평일 또는 주말을 쓰는 변수로 평일은 'weekday', 주말은 'weekend'를 쓰면 됩니다.
    * ok_bike_list : 원하는 자전거 정류소 번호를 리스트 형식으로 넣으면 됩니다. 예를 들면 [101,102,103,104,105]와 같은 형식입니다.
    * slot : 5 또는 24를 입력하면 됩니다.(int 형식으로)

  * fit 
    * clus_method : 'kmeans' 또는 'kmoids'를 쓰는 것으로 cluster 방법을 쓰면 됩니다.
    * cent_method : 초기값 지정 방법을 씁니다. 'density' 또는 'median'을 쓰면 됩니다.
    (주의 : clus_method에 'kmoids', cent_method에 'density'를 쓰면 오류가 나옵니다)
    * K1, K2 : 각각 지정할 K값을 쓰면 됩니다.

##### new_clustering.py
* 논문에서 제안한 클러스터 방법입니다.
* data 폴더에 서울시 정류소 정보, 자전거 대여기록(1시간 단위) 일부를 첨부했으며 여기 있는 파일 이름과 폴더명을 기준으로 사용됩니다.

##### kakao_local.py
* 다음카카오의 RESTAPI를 이용하는 방법입니다. 클러스터 방법론과는 별개로 다음카카오 RESTAPI 사용하시는데 참고하시면 될 것 같습니다. 코드를 여시면 rest_api 변수가 있는데 여기 키값을 입력하고 사용하시면 됩니다.

