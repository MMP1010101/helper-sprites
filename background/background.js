document.addEventListener('DOMContentLoaded', function() {
    // Variables para el fondo interactivo
    const body = document.body;
    
    // Efecto para que el fondo reaccione al movimiento del mouse
    body.addEventListener('mousemove', function(event) {
        const x = event.clientX / window.innerWidth;
        const y = event.clientY / window.innerHeight;
        
        // Calcular un leve cambio en la sombra interior según la posición del mouse
        const shadowX = 20 - (x * 40);
        const shadowY = 20 - (y * 40);
        
        // Ajustar la posición de la sombra para dar efecto de profundidad
        document.documentElement.style.setProperty('--shadow-x', `${shadowX}px`);
        document.documentElement.style.setProperty('--shadow-y', `${shadowY}px`);
    });
    
    // Función para inicializar elementos del juego
    function initGame() {
        console.log('Juego de cartas inicializado');
        // Aquí se pueden agregar más funcionalidades para inicializar el juego
    }
    
    // Inicializar el juego
    initGame();
});

// Función para generar un efecto visual cuando se juega una carta
function playCardEffect(element) {
    gsap.to(element, {
        scale: 1.2,
        rotation: Math.random() * 10 - 5,
        duration: 0.3,
        yoyo: true,
        repeat: 1,
        ease: "power2.out"
    });
}

// Función para distribuir cartas con animación
function dealCards(numCards, players) {
    return new Promise(resolve => {
        let delay = 0;
        for (let i = 0; i < numCards; i++) {
            for (let j = 0; j < players.length; j++) {
                setTimeout(() => {
                    console.log(`Carta repartida a jugador ${j+1}`);
                    // Aquí se añadiría la lógica de animación con GSAP
                }, delay);
                delay += 200;
            }
        }
        setTimeout(resolve, delay);
    });
}
