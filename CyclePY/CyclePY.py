import random

while True:
    print("\nVali:")
    print("Tunnitöö ülesanded: 1-7")
    print("Iseseisev töö ülesanded: 8, 9, 11, 12, 13")
    print("Iseseisvas töös puudub ülesanne 10, sest valisin 5 juhuslikku ülesannet.")
    valik = input("Sisesta ülesande number, 0 - väljumiseks: ")

    if valik == "1":
        print("Valisid 1. Sisestatakse 15 arvu. Määrata, mitu neist on täisarvud.")
        taisarvude_kogus = 0
        for i in range(15):
            try:
                arv = float(input(f"Sisesta {i+1}. arv: "))
                if arv.is_integer():
                    print(f"{arv} on täisarv.")
                    taisarvude_kogus += 1
                else:
                    print(f"{arv} ei ole täisarv.")
            except ValueError:
                print("Vale sisend! Palun sisesta arv.")
        print(f"Kokku sisestati {taisarvude_kogus} täisarvu.")

    elif valik == "2":
        print("Valisid 2. Küsi kasutajalt arv A ja leia kõigi naturaalarvude summa vahemikus 1 kuni A.")
        summa = 0
        while True:
            try:
                A = int(input("Sisesta naturaalarv A: "))
                if A < 1:
                    print("Vale sisend! Palun sisesta naturaalarv (suurem kui 0).")
                else:
                    for i in range(1, A + 1):
                        summa += i
                    print(f"Kõigi naturaalarvude summa vahemikus 1 kuni {A} on: {summa}")
                    break
            except ValueError:
                print("Palun sisesta täisarv.")

    elif valik == "3":
        print("Valisid 3. Sisestatakse 8 arvu. Leida nende korrutis (ainult positiivsete arvude puhul).")
        korrutis = 1
        on_positiivseid = False
        for i in range(8):
            try:
                arv = float(input(f"Sisesta {i+1}. arv: "))
                if arv > 0:
                    korrutis *= arv
                    on_positiivseid = True
                else:
                    print(f"{arv} ei ole positiivne arv.")
            except ValueError:
                print("Vale sisend! Palun sisesta arv.")
        if on_positiivseid:
            print(f"Positiivsete arvude korrutis on: {korrutis}")
        else:
            print("Positiivseid arve ei sisestatud.")

    elif valik == "4":
        print("Valisid 4. Väljasta ekraanile arvude ruudud vahemikus 10 kuni 20.")
        for i in range(10, 21):
            print(f"Arvu {i} ruut on: {i ** 2}")

    elif valik == "5":
        print("Valisid 5. Arvuta ainult negatiivsete arvude summa N sisestatud arvu seast.")
        while True:
            try:
                N = int(input("Sisesta arvude kogus N: "))
                if N < 1:
                    print("Vale sisend! Palun sisesta positiivne täisarv.")
                else:
                    negatiivsete_summa = 0
                    for i in range(N):
                        try:
                            arv = float(input(f"Sisesta {i+1}. arv: "))
                            if arv < 0:
                                negatiivsete_summa += arv
                            else:
                                print(f"{arv} ei ole negatiivne arv.")
                        except ValueError:
                            print("Vale sisend! Palun sisesta arv.")
                    print(f"Negatiivsete arvude summa on: {negatiivsete_summa}")
                    break
            except ValueError:
                print("VIGA! Palun sisesta täisarv.")

    elif valik == "6":
        print("Valisid 6. Määra sisestatud N arvu seast negatiivsete, positiivsete ja nullide arv.")
        while True:
            try:
                N = int(input("Sisesta arvude kogus N: "))
                if N < 1:
                    print("Vale sisend! Palun sisesta positiivne täisarv.")
                else:
                    negatiivsed = 0
                    positiivsed = 0
                    nullid = 0
                    for i in range(N):
                        try:
                            arv = float(input(f"Sisesta {i+1}. arv: "))
                            if arv < 0:
                                negatiivsed += 1
                            elif arv > 0:
                                positiivsed += 1
                            else:
                                nullid += 1
                        except ValueError:
                            print("Vale sisend! Palun sisesta arv.")
                    print(f"Negatiivseid arve: {negatiivsed}")
                    print(f"Positiivseid arve: {positiivsed}")
                    print(f"Nulle: {nullid}")
                    break
            except ValueError:
                print("VIGA! Palun sisesta täisarv.")

    elif valik == "7":
        print("Valisid 7. Väljasta ekraanile arvud, mis on K-ga jaguvad vahemikust [A, B].")
        while True:
            try:
                A = int(input("Sisesta vahemiku algus A: "))
                B = int(input("Sisesta vahemiku lõpp B: "))
                K = int(input("Sisesta jagaja K: "))
                if K == 0:
                    print("VIGA! Nulliga jagada ei saa.")
                elif A > B:
                    print("VIGA! A ei tohi olla suurem kui B.")
                else:
                    print(f"Arvud vahemikus [{A}, {B}], mis jaguvad arvuga {K}:")
                    leitud = False
                    for i in range(A, B + 1):
                        if i % K == 0:
                            print(i)
                            leitud = True
                    if not leitud:
                        print("Selliseid arve vahemikus ei leitud.")
                    break
            except ValueError:
                print("VIGA! Palun sisesta täisarv.")

    elif valik == "8":
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
        print("Valjumine...")
        break

    else:
        print("Vale valik!")
