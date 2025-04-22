document.getElementById("start-drawing").addEventListener("click", function () {
    fetch("/start_drawing")
        .then(response => response.json())
        .then(data => {
            alert(data.message);
        })
        .catch(error => console.error("Error:", error));
});
