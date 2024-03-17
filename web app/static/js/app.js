import React, { useState } from 'react';
import './static/css/styles.css';

function App() {
    const [topPlayers, setTopPlayers] = useState(null);
    const [optimizedPlayers, setOptimizedPlayers] = useState(null);

    const handlePredictions = () => {
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(response => response.json())
            .then(data => {
                setTopPlayers(data.top_players);
                setOptimizedPlayers(data.optimized_players);
            })
            .catch(error => {
                console.error('Error during prediction:', error);
            });
    };

    return (
        <div>
            // I will fill this in with my code
        </div>
    );
}

export default App;
