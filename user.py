class User:
    def __init__(
        self,
        id,
        position,
        area,
        a_liquid,
        group,
        cp,
        motive,
        fire_type,
        band,
        fire_date,
        pht,
        hire_date,
        antiquity_years,
        antiquity_days,
        genre,
        birth_place,
        classification,
        age,
        living_time,
        civil_status,
        children,
    ):
        self.id = id
        self.position = position
        self.area = area
        self.a_liquid = a_liquid
        self.group = group
        self.cp = cp
        self.motive = motive
        self.fire_type = fire_type
        self.band = band
        self.fire_date = fire_date
        self.pht = pht
        self.hire_date = hire_date
        self.antiquity_years = antiquity_years
        self.antiquity_days = antiquity_days
        self.genre = genre
        self.birth_place = birth_place
        self.classification = classification
        self.age = age
        self.living_time = living_time
        self.civil_status = civil_status
        self.children = children

    def to_dict(self):
        return {
            "ID": self.id,
            "Posición": self.position,
            "Area": self.area,
            "Á.liq.": self.a_liquid,
            "Grupo de personal": self.group,
            "CODIGO POSTAL": self.cp,
            "Motivo de la RENUNCIA": self.motive,
            "Tipo de Baja": self.fire_type,
            "Banda": self.band,
            "Baja": self.fire_date,
            "ReglaPHT": self.pht,
            "Alta": self.hire_date,
            "Antigüedad Clas": self.antiquity_years,
            "Antigüedad": self.antiquity_days,
            "Clave de sexo": self.genre,
            "Lugar de nacimiento": self.birth_place,
            "Clasificacion L. N": self.classification,
            "Edad del empleado": self.age,
            "¿Cuanto tiempo tiene viviendo en Cd. Juarez?": self.living_time,
            "Estado Civil": self.civil_status,
            "Hijos": self.children,
        }
