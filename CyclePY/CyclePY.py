import random

while True:
    print("\nVali")
    valik = input("Ulesanded 8-13 (valid 8,9,11,12,13(10 ei ole kuna valisin randomselt), 0 valjumiseks: ")

    if valik == "8":
        print("Valisid 8")
        for toll in range(1, 21):
            cm = toll * 2.5
            print(f"{toll} tolli = {cm} cm")

    elif valik == "9":
        print("Valisid 9")
        vigane = True
        while vigane:
            try:
                S = float(input("Sisesta algsumma S "))
                N = int(input("Sisesta aastate arv N "))
                if S <= 0 or N < 1:
                    print("Vale sisend")
                else:
                    for aasta in range(1, N + 1):
                        S = S * 1.03
                    print(f"Summa {N} aasta jarel on {round(S, 2)}")
                    vigane = False
            except ValueError:
                print("Viga sisesta arv")

    elif valik == "11":
        print("Valisid 11")
        juhuslik = random.randint(2, 50)
        print(f"Juhuslik arv on {juhuslik}")
        korrutis = 1
        leitud = False

        for arv in range(11, 100, 2):
            if arv % juhuslik == 0:
                korrutis *= arv
                leitud = True

        if leitud:
            print(f"Korrutis on {korrutis}")
        else:
            print("Selliseid arve ei leitud")

    elif valik == "12":
        print("Valisid 12")
        vigane = True
        while vigane:
            try:
                N = int(input("Sisesta niidukite arv N "))
                m = float(input("Mitu tundi tootas esimene niiduk "))
                if N < 1 or m <= 0:
                    print("Vale sisend")
                else:
                    kogu_aeg = 0
                    praegune_aeg = m
                    for i in range(N):
                        kogu_aeg += praegune_aeg
                        praegune_aeg += 10 / 60
                    print(f"Kogu aeg on {round(kogu_aeg, 2)} tundi")
                    vigane = False
            except ValueError:
                print("Viga sisesta arv")

    elif valik == "13":
        print("Valisid 13")
        kogus = 0
        summa = 0
        for i in range(100, 1001):
            if i % 7 == 0:
                print(i, end=" ")
                kogus += 1
                summa += i
        print(f"\nArve on kokku {kogus}")
        print(f"Summa on {summa}")

    elif valik == "0":
        print("Valjumine")
        break

    else:
        print("Vale valik")
