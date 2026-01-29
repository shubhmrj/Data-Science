# 🏥 Frontend Architecture & UI/UX Documentation

This document outlines the design philosophy, technical implementation, and features of the **Flask-Native Frontend** for the Heart Attack Prediction application. 

Unlike standard dashboards, this interface focuses on **clinical clarity**, **patient education**, and **responsive accessibility**.

---

## ⚡ Overview
This frontend is a server-side rendered (SSR) implementation using **Jinja2 templates**. It is designed to be lightweight, dependency-free (no Node.js/React required), and highly performant on mobile devices.

## 🎨 Design System: "Modern Clinical Glassmorphism"
We moved away from generic Bootstrap templates to a custom design language.
* **Color Palette:** Trustworthy Medical Blue (`#2563eb`), Deep Navy (`#1e3a8a`), and Soft Slate (`#f8fafc`).
* **Visual Style:** Glass-morphism cards with backdrop filters, subtle gradients, and high-fidelity SVG animations.
* **Typography:** `Plus Jakarta Sans` for clean, modern readability.

---

## 🚀 Key Features

### 1. The Intelligent Knowledge Hub 🧠
A dynamic side panel (on Desktop) or bottom-sheet (on Mobile) that acts as a real-time medical assistant.
* **Context Awareness:** As users click different fields (e.g., *Chest Pain*, *Slope*), the Hub updates instantly to explain what that medical term means.
* **Data Ranges:** Displays valid ranges (e.g., "Age: 29-77") based on the dataset.
* **Decision Support:** Provides "Healthy" vs "Risky" examples (e.g., *"0 Vessels = Blocked"* vs *"3 Vessels = Clear"*) to guide accurate data entry.

### 2. Mobile-First Responsiveness 📱
The layout adapts aggressively to screen size:
* **Desktop:** Split-screen view. Form on the left, Knowledge Hub fixed on the right.
* **Mobile:** Single-column form. The Knowledge Hub transforms into a **Sticky Bottom Sheet**, ensuring users can read guidance *while* typing without losing context.

### 3. Smart Risk Modals 🛡️
Instead of a simple text result, the app uses context-aware modals:
* **🔴 High Risk (Danger Mode):**
    * Triggers immediate visual warnings.
    * **Geolocation Integration:** Features a *"Locate Nearest Hospital"* button that uses the browser's GPS API to find cardiology centers via Google Maps.
    * **Safety Fallback:** If location is denied, it presents a prioritized "To-Do List" (e.g., *Stop exertion, Sit down*).
* **🟢 Low Risk (Safe Mode):**
    * Provides actionable lifestyle maintenance tips (Cardio, Diet, Hydration).

### 4. Inclusive & Robust Inputs 🤝
* **Gender Inclusivity:** The legacy dataset (1988) strictly requires binary inputs (0/1). We added an *"Other/Non-Binary"* option that politely explains this constraint and guides the user to select the biological/hormonal match for mathematical accuracy.
* **Input Validation:** Prevents negative numbers and unrealistic values.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Templating** | Jinja2 (Python/Flask) |
| **Styling** | CSS3 (Custom Properties / CSS Variables) |
| **Layout** | CSS Grid & Flexbox |
| **Interactivity** | Vanilla JavaScript (ES6+) |
| **Icons/Assets** | Inline SVGs (No external requests) |

---

## 📂 File Structure

```text
/templates
  ├── index.html      # Landing Page (SVG ECG Animation)
  └── home.html       # Main App (Knowledge Hub + Modals)
/static
  └── css
      └── style.css   # Single source of truth for all styles