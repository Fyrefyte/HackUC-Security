window.addEventListener("load", async () => {
    const Clerk = window.Clerk;
    if (!Clerk) {
        console.error("❌ Clerk not found.");
        return;
    }

    try {
        // Load Clerk
        await Clerk.load();
        console.log("✅ Clerk loaded successfully.");

        // Navbar elements
        const loginLink = document.getElementById("login-link");
        const rightLink = document.querySelector(".right-link"); // "Get Started"
        const getStarted = document.getElementById("get-started-btn");
        const logoutBtn = document.getElementById("logout-btn");
        const userGreeting = document.querySelector(".user-greeting");

        // Add a dashboard nav element (if not already in HTML)
        let dashboardLink = document.getElementById("dashboard-link");
        if (!dashboardLink) {
            dashboardLink = document.createElement("a");
            dashboardLink.id = "dashboard-link";
            dashboardLink.href = "dashboard.html";
            dashboardLink.textContent = "Dashboard";
            dashboardLink.style.display = "none"; // hidden by default
            // Append to your nav, adjust selector if needed
            const nav = document.querySelector(".left-links");
            if (nav) nav.appendChild(dashboardLink);
        }

        // Update navbar based on session
        const updateNavbar = async () => {
            const session = await Clerk.session;

            if (session && Clerk.user) {
                // User is signed in
                if (logoutBtn) logoutBtn.style.display = "inline-block";
                if (loginLink) loginLink.style.display = "none";
                if (rightLink) rightLink.style.display = "none";

                // Show greeting
                if (userGreeting) {
                    userGreeting.textContent = `Hello, ${Clerk.user.firstName}`;
                }

                // Show dashboard nav
                if (dashboardLink) dashboardLink.style.display = "inline-block";

            } else {
                // User is not signed in
                if (logoutBtn) logoutBtn.style.display = "none";
                if (loginLink) loginLink.style.display = "inline-block";
                if (rightLink) rightLink.style.display = "inline-block";

                // Clear greeting
                if (userGreeting) userGreeting.textContent = "";

                // Hide dashboard nav
                if (dashboardLink) dashboardLink.style.display = "none";
            }
        };

        // Initial navbar update
        updateNavbar();

        // Listen for user state changes
        Clerk.addListener(updateNavbar);

        // --- Login button triggers Clerk modal ---
        if (loginLink) {
            loginLink.addEventListener("click", async (e) => {
                e.preventDefault(); // prevent normal navigation
                await Clerk.openSignIn(); // open Clerk sign-in modal
            });
        }

        // --- Get Started button triggers Clerk modal ---
        if (rightLink) {
            rightLink.addEventListener("click", async (e) => {
                e.preventDefault();
                await Clerk.openSignUp(); // open Clerk sign-up modal
            });
        }

        if (getStarted) {
            getStarted.addEventListener("click", async (e) => {
                e.preventDefault();
                await Clerk.openSignUp(); // open Clerk sign-up modal
            });
        }

        // --- Logout button ---
        if (logoutBtn) {
            logoutBtn.addEventListener("click", async () => {
                await Clerk.signOut();
                window.location.href = "index.html";
                updateNavbar();
            });
        }

    } catch (err) {
        console.error("❌ Clerk failed to initialize:", err);
    }
});




