//// dashboard.js (MJPEG version)
window.addEventListener("load", async () => {
    const Clerk = window.Clerk;
    if (!Clerk) {
        console.error("❌ Clerk not found.");
        return;
    }

    try {
        await Clerk.load();
        console.log("✅ Clerk loaded successfully.");

        // Elements
        const greetingEl = document.querySelector(".user-greeting");
        const chatBox = document.getElementById("chat-box");
        const videoFeed = document.getElementById("video-feed"); // <img> for MJPEG
        const logoutBtn = document.getElementById("logout-btn");

        const updateDashboard = async () => {
            const session = await Clerk.session;
            const user = Clerk.user;

            if (session && user) {
                // Greeting
                if (greetingEl) {
                    greetingEl.textContent = `Hello, ${user.firstName}`;
                    greetingEl.style.display = "block";
                }

                // Show dashboard elements
                if (chatBox) chatBox.style.display = "block";
                if (videoFeed) videoFeed.style.display = "block";
                if (logoutBtn) logoutBtn.style.display = "inline-block";

                // MJPEG stream setup
                if (videoFeed) {
                    // Remove any previous placeholder
                    const existingPlaceholder = videoFeed.parentNode.querySelector(".video-placeholder");
                    if (existingPlaceholder) existingPlaceholder.remove();

                    // Set MJPEG source
                    videoFeed.src = "http://10.11.116.179:8000/video"; // MJPEG endpoint
                    videoFeed.alt = "Live Feed";

                    // Fallback if MJPEG fails
                    videoFeed.onerror = () => {
                        videoFeed.style.display = "none";

                        const placeholder = document.createElement("div");
                        placeholder.className = "video-placeholder";
                        placeholder.textContent = "No Live Feed";
                        placeholder.style.color = "white";
                        placeholder.style.fontSize = "1.5rem";
                        placeholder.style.fontStyle = "italic";
                        placeholder.style.textAlign = "center";
                        placeholder.style.backgroundColor = "#111";
                        placeholder.style.width = "100%";
                        placeholder.style.height = "100%";
                        placeholder.style.display = "flex";
                        placeholder.style.justifyContent = "center";
                        placeholder.style.alignItems = "center";

                        videoFeed.parentNode.appendChild(placeholder);
                    };
                }

                // Chat placeholder (read-only)
                if (chatBox) {
                    chatBox.innerHTML = ""; // Clear existing messages
                    setInterval(() => {
                        const msg = document.createElement("div");
                        msg.className = "chat-message";
                        msg.textContent = `Server: Placeholder message at ${new Date().toLocaleTimeString()}`;
                        chatBox.appendChild(msg);
                        chatBox.scrollTop = chatBox.scrollHeight; // auto-scroll
                    }, 3000);
                }

            } else {
                // Not logged in → hide dashboard & redirect
                if (greetingEl) greetingEl.style.display = "none";
                if (chatBox) chatBox.style.display = "none";
                if (videoFeed) videoFeed.style.display = "none";
                if (logoutBtn) logoutBtn.style.display = "none";

                window.location.href = "login.html";
            }
        };

        // Initial setup
        await updateDashboard();

        // Listen for login/logout changes
        Clerk.addListener(updateDashboard);

    } catch (err) {
        console.error("❌ Dashboard failed to initialize:", err);
    }
});

