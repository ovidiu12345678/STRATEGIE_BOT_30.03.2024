import requests
from threading import Timer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import openai

class StrategieMarkov(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, interval_timp: str, procent_balanta: float,
                 take_profit: float, stop_loss: float, alti_parametri: dict):
        super().__init__(client, contract, exchange, interval_timp, procent_balanta, take_profit, stop_loss, "Markov")
        self._volum_minim = alti_parametri.get('volum_minim', 0)
        self._numar_stari = 5
        self._matrice_tranzitie = self._estimeaza_matricea_tranzitie()
        self._stare_curenta = 1
        self._caracteristici = None
        self._etichete = None
        self.model_ml = RandomForestClassifier()
        self._cronometru_date_timp_real = Timer(60, self.colecteaza_date_timp_real)
        self._cronometru_date_timp_real.start()
        openai.api_key = 'cheia_openai_api'

    def colecteaza_date_timp_real(self):
        try:
            raspuns = requests.get("https://testnet.binancefuture.com/fapi/v1/ticker/bookTicker")
            if raspuns.status_code == 200:
                date = raspuns.json()
                if date is not None:
                    self._caracteristici, self._etichete = self.proceseaza_date_timp_real(date)
                    if self._caracteristici is not None:
                        self._antreneaza_model_ml()
        except Exception as e:
            print(f"Eroare la colectarea datelor în timp real: {e}")

        self._cronometru_date_timp_real = Timer(60, self.colecteaza_date_timp_real)
        self._cronometru_date_timp_real.start()

    def proceseaza_date_timp_real(self, date):
        caracteristici = []
        etichete = []

        if isinstance(date, list):
            for item in date:
                if 'price' in item:
                    pret = float(item['price'])
                    caracteristica = [pret]
                    caracteristici.append(caracteristica)
        elif isinstance(date, dict):
            if 'price' in date:
                pret = float(date['price'])
                caracteristica = [pret]
                caracteristici.append(caracteristica)
        else:
            print("Datele primite nu sunt în formatul așteptat.")
            return None, None

        return np.array(caracteristici), None

    def _estimeaza_matricea_tranzitie(self):
        contor_tranzitii = np.zeros((self._numar_stari, self._numar_stari))

        for i in range(1, len(self.lumanari)):
            inchidere_anterior = self.lumanari[i - 1].inchidere
            inchidere_curenta = self.lumanari[i].inchidere
            stare_anterioara = self._calculeaza_starea(inchidere_anterior)
            stare_curenta = self._calculeaza_starea(inchidere_curenta)
            contor_tranzitii[stare_anterioara - 1][stare_curenta - 1] += 1

        sume_rand = contor_tranzitii.sum(axis=1, keepdims=True)
        indici_zero_sume = np.where(sume_rand == 0)
        contor_tranzitii[indici_zero_sume] = 1
        matrice_tranzitie = np.divide(contor_tranzitii, sume_rand, out=np.zeros_like(contor_tranzitii),
                                      where=sume_rand != 0)

        return matrice_tranzitie

    def _calculeaza_starea(self, pret_inchidere: float) -> int:
        interval_pret = (self.lumanari[-1].maxim - self.lumanari[-1].minim) / self._numar_stari
        for i in range(1, self._numar_stari + 1):
            if pret_inchidere <= self.lumanari[-1].minim + i * interval_pret:
                return i
        return self._numar_stari

    def _calculeaza_volatilitatea(self) -> float:
        if len(self.lumanari) < 2:
            return 0

        maxime_lumanari = [lumanare.maxim for lumanare in self.lumanari]
        minime_lumanari = [lumanare.minim for lumanare in self.lumanari]
        interval_lumanari = max(maxime_lumanari) - min(minime_lumanari)
        pret_mediu_lumanari = sum((lumanare.inchidere + lumanare.deschidere) / 2 for lumanare in self.lumanari) / len(self.lumanari)
        volatilitate = interval_lumanari / pret_mediu_lumanari

        return volatilitate

    def _verifica_semnalul(self) -> int:
        volatilitate_curenta = self._calculeaza_volatilitatea()

        if volatilitate_curenta > 0.5:
            if self._profit_imediat_potențial() >= 0:
                if self._stare_curenta == 1:
                    if self.lumanari[-1].inchidere < self.lumanari[-2].minim:
                        self._stare_curenta = np.random.choice(range(1, self._numar_stari + 1),
                                                                p=self._matrice_tranzitie[self._stare_curenta - 1])
                elif self._stare_curenta == 5:
                    if self.lumanari[-1].inchidere > self.lumanari[-2].maxim:
                        self._stare_curenta = np.random.choice(range(1, self._numar_stari + 1),
                                                                p=self._matrice_tranzitie[self._stare_curenta - 1])
                else:
                    if self.lumanari[-1].inchidere < self.lumanari[-2].minim:
                        self._stare_curenta -= 1
                    elif self.lumanari[-1].inchidere > self.lumanari[-2].maxim:
                        self._stare_curenta += 1

                if self._stare_curenta >= 3:
                    return 1
                elif self._stare_curenta <= 1:
                    return -1
                else:
                    return 0
        else:
            return 0

    def _genereaza_predicția_gpt3(self, semnal: int) -> str:
        prompt = f"Sunt o strategie de tranzacționare financiară. Bazat pe analiza mea, am identificat un semnal de tranzacționare: {semnal}. Ar trebui să execut o tranzacție?"
        raspuns = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=50
        )
        if 'choices' in raspuns.data:
            return raspuns.data['choices'][0]['text'].strip()
        else:
            return "Nu se poate genera predicția."

    def _administreaza_pozitiile(self):
        if hasattr(self.pozitie_in_desfasurare, 'profit_nerealizat') and self.pozitie_in_desfasurare.profit_nerealizat > 0:
            while True:
                semnal = self._verifica_semnalul()
                profit_potential = self._profit_imediat_potențial()
                if semnal >= 0 and semnal >= profit_potential:
                    predictie = self._genereaza_predicția_gpt3(semnal)
                    print("Predictie GPT-3:", predictie)
                    if predictie.lower() == "da":
                        self._caracteristici = self._extrage_caracteristici()
                        self._etichete = self._genereaza_etichete()
                        self._antreneaza_model_ml()
                        semnal_ml = self._prezice_semnal_ml()
                        self._deschide_pozitie(semnal_ml)
                    break
                else:
                    time.sleep(5)

    def _extrage_caracteristici(self):
        caracteristici = []
        for lumanare in self.lumanari[-10:]:
            caracteristica = [lumanare.inchidere - lumanare.deschidere]
            caracteristica.append(np.mean([lumanare.inchidere for lumanare in self.lumanari[-5:]]))
            caracteristica.append(np.mean([lumanare.inchidere for lumanare in self.lumanari[-10:]]))
            caracteristici.append(caracteristica)
        return np.array(caracteristici)

    def _genereaza_etichete(self):
        etichete = []
        for i in range(len(self.lumanari[-10:])):
            if np.mean([lumanare.inchidere for lumanare in self.lumanari[-5:]]) > self.lumanari[-10:][i].inchidere:
                eticheta = 1
            else:
                eticheta = -1
            etichete.append(eticheta)
        return np.array(etichete)

    def _antreneaza_model_ml(self):
        if self._etichete is not None:
            X_antrenare, X_test, y_antrenare, y_test = train_test_split(self._caracteristici, self._etichete, test_size=0.2,
                                                                random_state=42)
            self.model_ml.fit(X_antrenare, y_antrenare)
            precizie = self.model_ml.score(X_test, y_test)
            print("Precizie Model:", precizie)

    def _prezice_semnal_ml(self):
        if self._caracteristici is not None:
            caracteristici = self._caracteristici[-1].reshape(1, -1)
            return self.model_ml.predict(caracteristici)
        else:
            return None

    def _profit_imediat_potențial(self):
        if not self.pozitie_in_desfasurare:
            return 0

        pret_curent = self.obtine_pretul_curent()
        if self.pozitie_in_desfasurare and self.pozitie_in_desfasurare.tip == "LUNG":
            return (pret_curent - self.pozitie_in_desfasurare.pret_intrare) / self.pozitie_in_desfasurare.pret_intrare
        elif self.pozitie_in_desfasurare and self.pozitie_in_desfasurare.tip == "SCURT":
            return (self.pozitie_in_desfasurare.pret_intrare - pret_curent) / self.pozitie_in_desfasurare.pret_intrare
        else:
            return 0

    def obtine_pretul_curent(self):
        try:
            raspuns = requests.get("https://testnet.binancefuture.com/fapi/v1/ticker/bookTicker")
            if raspuns.status_code == 200:
                date = raspuns.json()
                if date is not None:
                    pret_curent = float(date['price'])
                    return pret_curent
                else:
                    print("Eroare: Datele primite sunt nule.")
                    return None
            else:
                print("Eroare la obtinerea pretului curent:", raspuns.status_code)
                return None
        except Exception as e:
            print("Eroare la obtinerea pretului curent:", e)
            return None

    def verifica_tranzactie(self, tip_tick: str):
        if self.pozitie_in_desfasurare:
            if hasattr(self.pozitie_in_desfasurare, 'profit_nerealizat') and self.pozitie_in_desfasurare.profit_nerealizat >= 0:
                self._administreaza_pozitiile()
        else:
            self._deschide_pozitie_noua()

    def _deschide_pozitie_noua(self):
        
        while True:
            semnal = self._verifica_semnalul()  
            
            profit_potential = self._profit_imediat_potențial() 

           
            if semnal >= 0 and semnal >= profit_potential:
               
                self._caracteristici = self._extrage_caracteristici()
                self._etichete = self._genereaza_etichete()
                self._antreneaza_model_ml()
                semnal_ml = self._prezice_semnal_ml()

               
                if semnal_ml >= 0 and semnal_ml >= profit_potential:
                  
                    self._deschide_pozitie(semnal_ml)
                    break  

            time.sleep(5)  

    def _deschide_pozitie(self, semnal):
        
        pass
