function Enviar() {
    const fechaActual = obtenerFechaActual(); // Obtener la fecha actual
    const idi = document.getElementById('idi').value;
    const pos = document.getElementById('pos').value;
    const gp = document.getElementById('gp').value;
    const aliq = document.getElementById('aliq').value;
    const cp = document.getElementById('cp').value;
    const mr = document.getElementById('mr').value;
    const tb = document.getElementById('tb').value;
    const banda = document.getElementById('banda').value;
    const baja = document.getElementById('baja').value;
    const regla = document.getElementById('regla').value;
    const alta = document.getElementById('alta').value;
    const antclas = document.getElementById('antclas').value;
    const ant = document.getElementById('ant').value;
    const clave = document.getElementById('clave').value;
    const lugar = document.getElementById('lugar').value;
    const clas = document.getElementById('clas').value;
    const age = document.getElementById('age').value;
    const time = document.getElementById('time').value;
    const edocivil = document.getElementById('edocivil').value;
    const hijos = document.getElementById('hijos').value;

    const inventario = document.getElementById('inventario');

    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td>${fechaActual}</td>
        <td>${idi}</td>
        <td>${pos}</td>
        <td>${gp}</td>
        <td>${aliq}</td>
        <td>${cp}</td>
        <td>${mr}</td>
        <td>${tb}</td>
        <td>${banda}</td>
        <td>${baja}</td>
        <td>${regla}</td>
        <td>${alta}</td>
        <td>${antclas}</td>
        <td>${ant}</td>
        <td>${clave}</td>
        <td>${lugar}</td>
        <td>${clas}</td>
        <td>${age}</td>
        <td>${time}</td>
        <td>${edocivil}</td>
        <td>${hijos}</td>
    `;

    inventario.appendChild(newRow);

    // Enviar el producto a la hoja de cálculo
    enviarProductoHojaCalculo(fechaActual, idi, pos, gp, aliq, cp, mr, tb, banda, baja, regla, alta, antclas, ant, clave, lugar, clas, age, time, edocivil, hijos);

    alert("Producto añadido con éxito.");
}

async function enviarProductoHojaCalculo(fechaActual, idi, pos, gp, aliq, cp, mr, tb, banda, baja, regla, alta, antclas, ant, clave, lugar, clas, age, time, edocivil, hijos) {
    try {
        const respuesta = await fetch('https://sheet.best/api/sheets/1f79da48-9ec8-4116-8347-51c5e69a1763', {
            method: 'POST',
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "Fecha": fechaActual,
                "ID": idi,
                "Posición": pos,
                "Grupo de personal": gp,
                "Á.liq.": aliq,
                "Código Postal": cp,
                "Motivo de la RENUNCIA": mr,
                "Tipo de Baja": tb,
                "Banda": banda,
                "Baja": baja,
                "ReglaPHT": regla,
                "Alta": alta,
                "Antigüedad Clas": antclas,
                "Antigüedad": ant,
                "Clave de sexo": clave,
                "Lugar de nacimiento": lugar,
                "Clasificacion L. N": clas,
                "Edad del empleado": age,
                "¿Cuanto tiempo tiene viviendo en Cd. Juarez?": time,
                "Estado Civil": edocivil,
                "Hijos": hijos
            })
        });

        const contenido = await respuesta.json();
        console.log(contenido);
    } catch (error) {
        console.log(error);
    }
}
