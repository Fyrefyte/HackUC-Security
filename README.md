# HackUC-Security
# EchoGate

EchoGate is a web-based smart home security system that leverages **facial recognition** and **real-time notifications** to enhance home safety. This project includes a polished front-end website, a user dashboard for live monitoring, and authentication powered by **Clerk.dev**.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Folder Structure](#folder-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Technologies Used](#technologies-used)  
7. [Team](#team)  
8. [License](#license)  

---

## Project Overview
EchoGate is designed to provide homeowners with real-time surveillance and security control. It can:

- Recognize familiar faces and log them automatically.  
- Alert users immediately when an unknown individual is detected.  
- Send photos and live video feeds directly to the homeowner.  
- Provide an interface to remotely interact with visitors or notify authorities.

The system is built with a **responsive front-end**, live video streaming support, and secure user authentication.

---

## Features

### Home Page
- Responsive landing page with **hero section**, **project info**, and **team carousel**.  
- Interactive navigation including login, sign-up, and dashboard links.  
- Parallax effect for team image.  
- Team carousel with smooth sliding transitions.

### User Dashboard
- Live MJPEG video feed of your monitored area.  
- Real-time notifications in a chat-like interface (read-only).  
- Personalized greeting for logged-in users.  
- Dashboard access restricted to authenticated users.  

### Authentication
- Full integration with [**Clerk.dev**](https://clerk.dev) for secure login, sign-up, and session management.  
- Conditional display of navigation links and buttons depending on user authentication state.

---

## Folder Structure
/EchoGate
│
├─ index.html # Home page
├─ dashboard.html # User dashboard
├─ login.html # Login page
├─ signup.html # Sign-up page
├─ style.css # Global CSS
├─ script.js # Home page JS (carousel & parallax)
├─ clerk-dev.js # Clerk.dev authentication JS
├─ dashboard.js # Dashboard functionality JS
├─ images/ # All images including logos & team photos
└─ README.md # Project documentation

yaml
Copy code

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/EchoGate.git
cd EchoGate
Open index.html in a web browser to view the home page.

For dashboard functionality:

Make sure MJPEG streaming server is running at the configured URL (default: http://10.11.116.179:8000/video).

Open dashboard.html in a browser.

Clerk.dev authentication requires a publishable key configured in your HTML. Replace with your own key if needed.

Usage
Use the Get Started button to sign up.

Use Login for returning users.

After logging in, access the Dashboard to view live video feed and notifications.

The parallax effect on the home page and carousel for team members are interactive with scroll and arrow controls.

Technologies Used
HTML5 & CSS3 for responsive design.

JavaScript (ES6) for DOM manipulation and interactivity.

Clerk.dev for authentication and session management.

MJPEG streaming for live video feed.

Parallax effect and carousel implemented with vanilla JS.

Team
Leo Canales – Facial recognition & motion detection systems.

Zander Avery – Front-end development & website functionality.

Eric Ngo – Logo design & video presentation.

Anirudh Chari – Logo design & video scripting.

License
This project is © EchoGate, 2025.
All rights reserved. Designed with safety and security in mind.
