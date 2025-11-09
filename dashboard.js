// dashboard.js
window.addEventListener("load", async () => {
    const Clerk = window.Clerk;
    if (!Clerk) {
        console.error("❌ Clerk not found.");
        return;
    }

    try {
        await Clerk.load();
        console.log("✅ Clerk loaded successfully.");

        const greetingEl = document.querySelector(".user-greeting");
        const videoFeed = document.getElementById("video-feed");

        const updateDashboard = async () => {
            const session = await Clerk.session;
            const user = Clerk.user;

            if (!(session && user)) {
                // Not logged in → redirect to login
                if (greetingEl) greetingEl.style.display = "none";
                if (videoFeed) videoFeed.style.display = "none";
                window.location.href = "login.html";
                return;
            }

            // Show greeting
            if (greetingEl) {
                greetingEl.textContent = `Hello, ${user.firstName}`;
                greetingEl.style.display = "block";
            }

            // MJPEG video feed
            if (videoFeed) {
                videoFeed.src = "http://10.11.116.179:8000/video"; // MJPEG feed
                videoFeed.alt = "Live Feed";

                videoFeed.onerror = () => {
                    videoFeed.style.display = "none";

                    const ph = document.createElement("div");
                    ph.className = "video-placeholder";
                    ph.textContent = "No Live Feed";
                    ph.style.cssText = `
                        color: white;
                        font-size: 1.5rem;
                        font-style: italic;
                        text-align: center;
                        background-color: #111;
                        width: 100%;
                        height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    `;
                    videoFeed.parentNode.appendChild(ph);
                };
            }
        };

        await updateDashboard();
        Clerk.addListener(updateDashboard);

    } catch (err) {
        console.error("❌ Dashboard failed to initialize:", err);
    }
});

