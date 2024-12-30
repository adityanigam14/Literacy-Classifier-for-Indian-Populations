document.getElementById("prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    // Collect the form data
    const formData = {
        social_group: document.getElementById("social_group").value,
        rural_urban: document.getElementById("rural_urban").value,
        state: document.getElementById("state").value,
        gender: document.getElementById("gender").value,
        age: parseInt(document.getElementById("age").value),
        internet_access: document.getElementById("internet_access").value,
        computer_access: document.getElementById("computer_access").value,
        marital_status: document.getElementById("marital_status").value
    };

    try {
        // Call the FastAPI endpoint
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(formData),
        });

        // Parse the response
        const result = await response.json();
        console.log(result); // Add this line to debug

        if (response.ok) {
            document.getElementById("result").innerText = `This individual is predicted to be: ${result.status}`;
        } else {
            document.getElementById("result").innerText = `Error: ${result.detail}`;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error connecting to the server.";
    }
});
