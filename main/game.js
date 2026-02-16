// Configuración principal del juego de cartas
document.addEventListener('DOMContentLoaded', function() {
    // Constantes del juego
    const CARD_WIDTH = 100;
    const CARD_HEIGHT = 140;
    
    // Estado del juego
    let gameState = {
        deck: [],
        players: [],
        currentPlayer: 0,
        isGameActive: false
    };
    
    // Inicializar el juego
    function initGame() {
        console.log("Inicializando juego Kartaska");
        
        // Crear baraja
        createDeck();
        
        // Configurar jugadores
        setupPlayers();
        
        // Iniciar juego
        startGame();
    }
    
    // Crear baraja de cartas
    function createDeck() {
        // Usar la librería deck-of-cards para crear un mazo
        gameState.deck = Deck();
        
        console.log("Baraja creada");
    }
    
    // Configurar jugadores
    function setupPlayers() {
        // Crear jugador principal
        gameState.players.push({
            id: 'player',
            name: 'Jugador',
            hand: [],
            score: 0
        });
        
        // Crear oponentes (CPU)
        for (let i = 0; i < 3; i++) {
            gameState.players.push({
                id: `cpu-${i}`,
                name: `CPU ${i+1}`,
                hand: [],
                score: 0,
                isComputer: true
            });
        }
        
        console.log("Jugadores configurados:", gameState.players.length);
    }
    
    // Iniciar la partida
    function startGame() {
        console.log("¡Comenzando partida!");
        gameState.isGameActive = true;
        
        // Barajar cartas
        shuffleDeck();
        
        // Repartir cartas
        dealCardsToPlayers();
        
        // Renderizar interfaz
        renderGameUI();
    }
    
    // Barajar la baraja
    function shuffleDeck() {
        gameState.deck.shuffle();
        console.log("Baraja mezclada");
    }
    
    // Repartir cartas a los jugadores
    function dealCardsToPlayers() {
        // Utilizar la función de animación del archivo background.js
        dealCards(5, gameState.players).then(() => {
            console.log("Todas las cartas repartidas");
        });
    }
    
    // Renderizar interfaz del juego
    function renderGameUI() {
        // Aquí iría el código para renderizar las cartas y elementos visuales
        console.log("Interfaz renderizada");
    }
    
    // Iniciar el juego automáticamente
    setTimeout(initGame, 1000);
});
