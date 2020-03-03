# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:20:50 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:35:23 2019

@author: User
"""

"""
참고사이트
https://stackoverflow.com/questions/49456385/running-flask-from-ipython-raises-systemexit
https://medium.com/@dydrlaks/flask-%EC%B9%B4%EC%B9%B4%EC%98%A4-%EC%82%AC%EC%9A%A9%EC%9E%90%EA%B4%80%EB%A6%AC-rest-api-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-e07ff5aff018
http://blog.daum.net/_blog/BlogTypeView.do?blogid=0drsH&articleno=79&_bloghome_menu=recenttext&totalcnt=77
https://dong1lkim.oboki.net/workspace/programming/language/python/kakao-api-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%A3%BC%EC%86%8C-%EC%A2%8C%ED%91%9C-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0/
https://a1010100z.tistory.com/entry/Spring-RESTFul-API-GET%EC%9C%BC%EB%A1%9C-JSON-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B0%9B%EC%95%84%EC%99%80-%ED%8C%8C%EC%8B%B1%EA%B9%8C%EC%A7%80JAVA-Spring-Kakao-%EB%A1%9C%EC%BB%AC-API-%EC%A1%B8%EC%97%85%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B83
https://developers.kakao.com/docs/restapi/local#%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC-%EA%B2%80%EC%83%89
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:53:01 2019

@author: User
"""

import requests
import json
import numpy as np
import pandas as pd

# kakao에서 받아온 rest_api key => KaKaoAppkey에 넣어야 함.


def neighbor_category_search(x_list, y_list, radius, category):
    """
    
    *** (x,y) 좌표를 중심으로 반경 내에 특정 카테고리에 속하는 것의 개수를 반환함. ***
    
    x_list : x 좌표의 list
    y_list : y 좌표의 list
    radius : 반경 길이(m)
    category : 카테고리 코드(아래 코드 참조)
    MT1 : 대형마트       |    CS2 : 편의점    |  PS3 : 어린이집, 유치원
    SC4 : 학교           |    AC5 : 학원      | PK6 : 주차장
    OL7 : 주유소, 충전소 |    SW8 : 지하철역   | BK9 : 은행
    CT1 : 문화시설       |    AG2 : 중개업소  |  PO3 : 공공기관
    AT4 : 관광명소       |    AD5 : 숙박      | FD6 : 음식점
    CE7 : 카페           |    HP8 : 병원      | PM9 : 약국
    """
    
    # n : search 할 총 좌표 수
    # count : 해당 카테고리의 총 개수
    # count_list : 카테고리 개수 리스트
    # url : 바꿀 주소.
    n = len(x_list)
    count_list= []
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        
        
        url = 'https://dapi.kakao.com/v2/local/search/category.json?category_group_code='+str(category)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)
        result = json.loads(requests.get(url,headers=headers).text)
        
        count = result['meta']['total_count']
        count_list.append(count)

    return count_list



def neighbor_keyword_search(x_list, y_list, radius, keyword):
    """
    *** (x,y) 좌표를 중심으로 반경 내에 특정 키워드가 들어가는 곳을 찾음.
    
    x_list : x 좌표의 list
    y_list : y 좌표의 list
    radius : 반경 길이(m)
    keyword : 키워드
    """

    # n : search 할 총 좌표 수
    # count : 해당 카테고리의 총 개수
    # count_list : 카테고리 개수 리스트
    # url : 바꿀 주소.
    
    n = len(x_list)
    count_list= []
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)
        result = json.loads(requests.get(url,headers=headers).text)
        
        count = result['meta']['total_count']
        count_list.append(count)
        
    return count_list




def neighbor_keyword_search_dict(sta_num_list, x_list, y_list, radius, keyword, c_group_code='None'):
    """
    *** (x,y) 좌표를 중심으로 반경 내에 특정 키워드가 들어가는 곳을 찾음.
        결과값은 dict로 제공함.
    
    sta_num_list : 자전거 대여소 정류장 번호.
    x_list : x 좌표의 list
    y_list : y 좌표의 list
    radius : 반경 길이(m)
    keyword : 키워드
    """
    
    # n : search 할 총 좌표 수
    # station_dict : 각 따릉이 정류소 번호랑 연결되는 주소값.
    # url : 바꿀 주소.
    
    station_dict = {}   
    n = len(x_list)
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        if c_group_code == 'None':
            url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)
        else :
            url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&category_group_code='+c_group_code+'&radius='+str(radius)
            
        result = json.loads(requests.get(url,headers=headers).text)
        
        # 각 정류소에 해당하는 위치이름, 주소, 거리값을 넣을 리스트
        place_list = []
        address_list = []
        dist_list = []
        id_list = []
        
        page_nums = int((result['meta']['total_count']/15))+1

        for page_num in range(1,page_nums+1) :

            url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&category_group_code='+c_group_code+'&radius='+str(radius)+'&sort=distance&page='+str(page_num)
            result = json.loads(requests.get(url,headers=headers).text)
            each_result = result['documents']
            for j in each_result :

                # place_name : 장소 이름
                # address : 주소(도로명)
                # dist : 원 좌표에서의 거리
                place_name = j['place_name']
                address = j['road_address_name']
                dist = j['distance']
                ids = j['id']
                
                # 각 리스트에 추가함.
                place_list.append(place_name)
                address_list.append(address)
                dist_list.append(dist)
                id_list.append(ids)
        # 각 값을 추가함.
        station_dict[sta_num_list[i]] = {'place' : place_list, 'address' : address_list, 'dist' : np.array(dist_list, dtype='int'), 'id':id_list}
        
        print(i)
    return station_dict



#neighbor_category_search_dist([1925], [37.49989], [126.8677], 10000, 'CT1')


def neighbor_category_search_dist(sta_num_list, x_list, y_list, radius, category):
    """
    
    *** (x,y) 좌표를 중심으로 반경 내에 특정 카테고리에 속하는 것의 dict값을 반환함. ***
    
    x_list : x 좌표의 list
    y_list : y 좌표의 list
    radius : 반경 길이(m)
    category : 카테고리 코드(아래 코드 참조)
    MT1 : 대형마트       |    CS2 : 편의점    |  PS3 : 어린이집, 유치원
    SC4 : 학교           |    AC5 : 학원      | PK6 : 주차장
    OL7 : 주유소, 충전소 |    SW8 : 지하철역   | BK9 : 은행
    CT1 : 문화시설       |    AG2 : 중개업소  |  PO3 : 공공기관
    AT4 : 관광명소       |    AD5 : 숙박      | FD6 : 음식점
    CE7 : 카페           |    HP8 : 병원      | PM9 : 약국
    """
    
    # n : search 할 총 좌표 수
    # count : 해당 카테고리의 총 개수
    # count_list : 카테고리 개수 리스트
    # url : 바꿀 주소.
    
    station_dict = {}
    n = len(x_list)
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        
        
        url = 'https://dapi.kakao.com/v2/local/search/category.json?category_group_code='+str(category)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)
        
        result = json.loads(requests.get(url,headers=headers).text)
        
        # 각 정류소에 해당하는 위치이름, 주소, 거리값을 넣을 리스트
        place_list = []
        address_list = []
        dist_list = []
        id_list = []
        
        # result_docu : 검색된 각각의 결과값.
        page_nums = int((result['meta']['total_count']/15))+1

        for page_num in range(1,page_nums+1) :
        
            url = 'https://dapi.kakao.com/v2/local/search/category.json?category_group_code='+str(category)+'&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)+'&sort=distance&page='+str(page_num)
            result = json.loads(requests.get(url,headers=headers).text)
            each_result = result['documents']
            for j in each_result :
            
                # place_name : 장소 이름
                # address : 주소(도로명)
                # dist : 원 좌표에서의 거리
                place_name = j['place_name']
                address = j['road_address_name']
                dist = j['distance']
                ids = j['id']
                
                # 각 리스트에 추가함.
                place_list.append(place_name)
                address_list.append(address)
                dist_list.append(dist)
                id_list.append(ids)
            # 각 값을 추가함.
            
            if result['meta']['is_end'] == True:
                break
        station_dict[sta_num_list[i]] = {'place' : place_list, 'address' : address_list, 'dist' : np.array(dist_list, dtype='int'), 'id':id_list}
        print(i)
        
    return station_dict




def address_to_coordinate(school_name_list, address_list):
    """
    *** 입력한 주소의 좌표값을 제공함.
    
    school_name_list : 학교 이름 리스트.
    address_list : 주소값 리스트.
    """
    
    # n : search 할 총 좌표 수
    # station_dict : 각 따릉이 정류소 번호랑 연결되는 주소값.
    # url : 바꿀 주소.
    
    address_dict = {}   
    n = len(address_list)
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        url = 'https://dapi.kakao.com/v2/local/search/address.json?query='+str(address_list[i])
            
        result = json.loads(requests.get(url,headers=headers).text)
        
        # 각 정류소에 해당하는 위치이름, 주소, 거리값을 넣을 리스트
        x_list = []
        y_list = []
        
        page_nums = int((result['meta']['total_count']/15))+1

        for page_num in range(1,page_nums+1) :

            url = 'https://dapi.kakao.com/v2/local/search/address.json?query='+str(address_list[i])
            result = json.loads(requests.get(url,headers=headers).text)
            
            each_result = result['documents']
            for j in each_result :

                # place_name : 장소 이름
                # address : 주소(도로명)
                # dist : 원 좌표에서의 거리
                x_coor = j['address']['y']
                y_coor = j['address']['x']
                
                # 각 리스트에 추가함.
                x_list.append(x_coor)
                y_list.append(y_coor)

        # 각 값을 추가함.
        address_dict[i] = {'name' : school_name_list[i], 'x_coor' : x_coor, 'y_coor' : y_coor}
        
        print(i)
    return address_dict


def coordinate_to_address(x_list, y_list):
    """
    *** 입력한 주소의 좌표값을 제공함.
    
    x_list : x좌표.
    y_list : y좌표.
    """
    
    # n : search 할 총 좌표 수
    # station_dict : 각 따릉이 정류소 번호랑 연결되는 주소값.
    # url : 바꿀 주소.
    
    address_dict = {}   
    n = len(x_list)
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        
        x_coor = x_list[i]
        y_coor = y_list[i]
        url = 'https://dapi.kakao.com/v2/local/geo/coord2address.json?x='+str(y_coor)+'&y='+str(x_coor)
        
        result = json.loads(requests.get(url,headers=headers).text)
        try : 
            address = result['documents'][0]['address']['address_name']
        except(IndexError):
            address = ''
            state_name = ''
            address_dict[i] = {'address' : address, 'state':state_name, 'x_coor' : x_coor, 'y_coor' : y_coor}
            continue
        state_name = address.split(' ')[1]
        # 각 정류소에 해당하는 위치이름, 주소, 거리값을 넣을 리스트
        
        
        # 각 값을 추가함.
        address_dict[i] = {'address' : address, 'state':state_name, 'x_coor' : x_coor, 'y_coor' : y_coor}
        
        print(i)
    return address_dict



def keyword_to_coordinate(school_name_list, keyword_list):
    """
    *** 입력한 주소의 좌표값을 제공함.
    
    school_name_list : 학교 이름 리스트.
    address_list : 주소값 리스트.
    """
    
    # n : search 할 총 좌표 수
    # station_dict : 각 따릉이 정류소 번호랑 연결되는 주소값.
    # url : 바꿀 주소.
    
    keyword_dict = {}   
    n = len(keyword_list)
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword_list[i])
            
        result = json.loads(requests.get(url,headers=headers).text)
        
        # 각 정류소에 해당하는 위치이름, 주소, 거리값을 넣을 리스트
        x_list = []
        y_list = []
        
        page_nums = int((result['meta']['total_count']/15))+1

        for page_num in range(1,page_nums+1) :

            url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(keyword_list[i])
            result = json.loads(requests.get(url,headers=headers).text)
            
            each_result = result['documents']
            for j in each_result :

                # place_name : 장소 이름
                # address : 주소(도로명)
                # dist : 원 좌표에서의 거리
                x_coor = j['y']
                y_coor = j['x']
                
                # 각 리스트에 추가함.
                x_list.append(x_coor)
                y_list.append(y_coor)

        # 각 값을 추가함.
        keyword_dict[i] = {'name' : school_name_list[i], 'x_coor' : x_list, 'y_coor' : y_coor, 'result':result}
        
        print(i)
    return keyword_dict



 
def subway_sta_search(subway_df):
    """
    *** 지하철역 이름과 매칭되는 좌표값을 구함 ***
    
    subway_df : 지하철역 이름, 호선 이름 있는 데이터프레임
    
    """
    
    # n : search 할 총 좌표 수
    # count : 해당 카테고리의 총 개수
    # count_list : 카테고리 개수 리스트
    # url : 바꿀 주소.
    
    n = len(subway_df)
    rest_api = ''
    for i in range(n):
        search_sta_name = subway_df.loc[i,'전철역명']
        search_sta_line = subway_df.loc[i,'호선']
        
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+str(search_sta_name)+'역&category_group_code=SW8'
        result = json.loads(requests.get(url,headers=headers).text)
        result_docu = result['documents']
        
        print(i)
        
        for j in result_docu :
            result_list = j['place_name'].split(' ')
            
            if len(result_list) == 0:
                break
            
            if result_list[0] == '봉화산역6호선':
                result_list = j['place_name'].split('역')
            
            sta_name = result_list[0].strip()
            sta_line = result_list[1].strip()
            
            if (sta_name.rstrip('역') == search_sta_name.rstrip('역')) & (sta_line == search_sta_line) :
                subway_df.loc[i,'long'] = j['x']
                subway_df.loc[i,'lati'] = j['y']
                break
    
    return subway_df





def search_radius_subway(station_list, x_list, y_list, radius, subway_count_df):
    
    """
    *** 특정 좌표의 반경 안에 있는 지하철역들의 호선, 이름을 구함. ***
    station_list : 자전거 정류소 번호
    
    x_list : x 좌표의 list
    y_list : y 좌표의 list
    radius : 반경 길이(m)
    subway_count_df : 특정 역들에 대해 각 시간대의 승객수가 있는 데이터.
    
    """

    # n : search 할 총 좌표 수
    # name_line_list : dict 형태로 name, line 데이터 저장.
    
    n = len(x_list)
    name_line_list = []
    rest_api = ''
    for i in range(n):
        
        headers = ({'Authorization' : "KakaoAK "+rest_api})
        
                
        url = 'https://dapi.kakao.com/v2/local/search/category.json?category_group_code=SW8&x='+str(y_list[i])+'&y='+str(x_list[i])+'&radius='+str(radius)
        result = json.loads(requests.get(url,headers=headers).text)
                     
        # 각 이름, 호선 값을 추출하여 리스트 형태로 넣을 list
        temp_name_list = []
        temp_line_list = []
        
        # 데이터가 검색 되었는지 찾아냄
        if result['meta']['total_count'] != 0:
            result_docu = result['documents']


            # result_docu : 역 개수만큼 각각의 정류소가 dict 형태로 저장됨
            for j in result_docu :
                
                # '역이름 호선'으로 데이터가 구성됨
                result_list = j['place_name'].split(' ')
                
                # 만약 결과값 없으면 바로 중단.
                if len(result_list) == 0:
                    break
                
                # 봉화산역의 경우 붙어서 결과가 나오므로 다르게 만듦.
                if result_list[0] == '봉화산역6호선':
                    result_list = j['place_name'].split('역')
                    
                sta_name = result_list[0].strip()
                sta_line = result_list[1].strip()
                
                # temp_name_list : 각 좌표에 대해 나온 지하철 정류소 이름을 리스트 형태로 저장함
                    # 역이란 단어를 없앰.
                # temp_line_list : 각 좌표에 대해 나온 지하철 호선 이름을 리스트 형태로 저장함
                temp_name_list.append(sta_name.rstrip('역'))
                temp_line_list.append(sta_line)
            
        name_line_list.append({'bike_sta': station_list[i], 'subway_name' : temp_name_list, 'subway_line' : temp_line_list})
            
        print(i)
        
    return name_line_list
    
    
