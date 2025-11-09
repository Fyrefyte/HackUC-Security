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
        const logoutBtn = document.getElementById("logout-btn");
        const middleGreeting = document.querySelector(".middle-greeting");

        // Update navbar based on session
        const updateNavbar = async () => {
            const session = await Clerk.session;

            if (session && Clerk.user) {
                // User is signed in
                if (logoutBtn) logoutBtn.style.display = "inline-block";
                if (loginLink) loginLink.style.display = "none";
                if (rightLink) rightLink.style.display = "none";

                // Show greeting
                if (middleGreeting) {
                    middleGreeting.textContent = `Hello, ${Clerk.user.firstName}`;
                }
            } else {
                // User is not signed in
                if (logoutBtn) logoutBtn.style.display = "none";
                if (loginLink) loginLink.style.display = "inline-block";
                if (rightLink) rightLink.style.display = "inline-block";

                // Clear greeting
                if (middleGreeting) middleGreeting.textContent = "";
            }
        };

        // Initial navbar update
        updateNavbar();

        // Listen for user state changes
        Clerk.addListener(updateNavbar);

        // Logout button
        if (logoutBtn) {
            logoutBtn.addEventListener("click", async () => {
                await Clerk.signOut();
                updateNavbar();
            });
        }

    } catch (err) {
        console.error("❌ Clerk failed to initialize:", err);
    }
});



