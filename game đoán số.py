import random

so_lan_traloi = 0
so_random = random.randint(1, 50)

print("Hãy chọn 1 số ngẫu nhiên từ 1 đến 50 \r")

while so_lan_traloi < 3:
    so_lan_traloi += 1
    doan = input("Con số dự đoán của bạn lần thứ "+ str(so_lan_traloi)+ " là:" )
    doan = int(doan)
    
    if doan < so_random:
        print("Số bạn đoán nhỏ hơn số ngẫu nhiên!")
    elif doan > so_random:
        print("Số bạn đoán lớn hơn số ngẫu nhiên!")
    else:
        break
    
if doan == so_random:
    print("Chúc mừng bạn đã đoán đúng!!!")
else:
    print("Bạn đã đoán sai, con số chính xác là " + str(so_random))