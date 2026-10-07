"""
Agentic AI Service using Google Gemini API with function calling.
The agent orchestrates: weather lookup, crop recommendation, yield prediction,
crop soil requirements (N, P, K, pH), location-aware better crop recommendations,
and agricultural advisory generation.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai

from .ml_service import ml_service
from .weather_service import get_weather_for_location

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Agent tool schemas for Gemini function calling
AGENT_TOOLS = [
    {
        "name": "get_crop_requirements",
        "description": (
            "Get the exact ideal soil pH range, Nitrogen (N in kg/ha), Phosphorus (P in kg/ha), "
            "Potassium (K in kg/ha) requirements, temperature, humidity, and rainfall ranges "
            "for any asked crop (e.g., Rice, Wheat, Cotton, Maize, Banana, Coffee, etc.). "
            "Always use this when the user asks about soil requirements, pH, or NPK for a crop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {
                    "type": "string",
                    "description": "Name of the crop (e.g. Rice, Wheat, Cotton, Maize, Banana)",
                },
            },
            "required": ["crop"],
        },
    },
    {
        "name": "get_location_soil_profile",
        "description": (
            "Look up the empirical regional soil Nitrogen (N), Phosphorus (P), Potassium (K), "
            "and pH values for any Indian state or district (e.g. Punjab, Maharashtra, Tamil Nadu, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Name of the Indian state or city (e.g. Punjab, Ludhiana, Maharashtra, Tamil Nadu)",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "recommend_better_crops_for_location",
        "description": (
            "Recommend the optimal/better crops for a given location or state in India. "
            "Automatically retrieves regional soil chemistry (N, P, K, pH), local weather, "
            "executes the trained ML Recommendation Classifier, and forecasts yields with XGBoost. "
            "If current_crop is provided, it performs a head-to-head suitability and yield comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Indian state or city (e.g. Punjab, Ludhiana, Maharashtra, Coimbatore, Tamil Nadu)",
                },
                "current_crop": {
                    "type": "string",
                    "description": "Optional crop the farmer is currently considering or asking about (e.g. Rice, Wheat)",
                },
                "season": {
                    "type": "string",
                    "description": "Optional season (Kharif, Rabi, Whole Year, Summer)",
                },
                "land_area_ha": {
                    "type": "number",
                    "description": "Optional farm area in hectares (default 1.0)",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Fetch current weather conditions and annual climate estimates for a given "
            "latitude/longitude. Returns temperature, humidity, rainfall."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude":  {"type": "number", "description": "Latitude of location"},
                "longitude": {"type": "number", "description": "Longitude of location"},
                "location_name": {"type": "string", "description": "Human-readable location name"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "predict_yield",
        "description": (
            "Predict crop yield (tonnes per hectare) for given crop, location, soil, and weather. "
            "Uses the trained ML regression model. Never invents yield values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "crop":     {"type": "string"},
                "state":    {"type": "string"},
                "season":   {"type": "string"},
                "area":     {"type": "number", "description": "Area in hectares"},
                "fertilizer": {"type": "number", "description": "Total fertilizer kg"},
                "pesticide": {"type": "number", "description": "Pesticide kg"},
                "year":     {"type": "integer"},
                "N":        {"type": "number", "description": "Soil Nitrogen"},
                "P":        {"type": "number", "description": "Soil Phosphorus"},
                "K":        {"type": "number", "description": "Soil Potassium"},
                "pH":       {"type": "number"},
                "avg_temp_c": {"type": "number"},
                "total_rainfall_mm": {"type": "number"},
                "avg_humidity_percent": {"type": "number"},
            },
            "required": ["crop", "state", "season", "area"],
        },
    },
    {
        "name": "recommend_crop",
        "description": (
            "Recommend the most suitable crop given specific custom N, P, K, pH and weather conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "N":           {"type": "number"},
                "P":           {"type": "number"},
                "K":           {"type": "number"},
                "ph":          {"type": "number"},
                "temperature": {"type": "number"},
                "humidity":    {"type": "number"},
                "rainfall":    {"type": "number"},
                "state":       {"type": "string"},
            },
            "required": ["N", "P", "K", "ph"],
        },
    },
    {
        "name": "compare_crops",
        "description": (
            "Compare two or more crops by predicting yield for each under the same regional conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "crops":  {"type": "array", "items": {"type": "string"}},
                "state":  {"type": "string"},
                "season": {"type": "string"},
                "area":   {"type": "number"},
                "year":   {"type": "integer"},
                "avg_temp_c": {"type": "number"},
                "total_rainfall_mm": {"type": "number"},
                "avg_humidity_percent": {"type": "number"},
            },
            "required": ["crops", "state"],
        },
    },
    {
        "name": "get_known_options",
        "description": "Returns lists of known crops, states, and seasons supported by the ML models.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """
You are an expert Agricultural Decision AI and Agronomist for Indian farmers and researchers.
Your primary goals:
1. Provide realistic, scientifically grounded soil parameters (Soil pH, Nitrogen N, Phosphorus P, Potassium K) for any requested crop.
2. Recommend the best/optimal crops for any requested location (state or district in India) using our trained Machine Learning models (Random Forest classifier & XGBoost yield engine).
3. If a farmer asks about a specific crop in a location (e.g. "I want to grow rice in Punjab, is there a better crop?"), provide a clear comparison between their intended crop and the ML-recommended better crops, explaining soil chemistry, water requirements, and estimated yield.

IMPORTANT RULES & GUIDELINES:
- When asked for soil pH and NPK for a crop: ALWAYS execute `get_crop_requirements` to fetch the real empirical ranges. Present the optimal pH, N, P, K in kg/ha, temperature, rainfall, and practical fertilizer advice.
- When asked for crop recommendation for a location: ALWAYS execute `recommend_better_crops_for_location` or `get_location_soil_profile` to use the actual regional soil data.
- NEVER fabricate numbers; use the exact results returned by the tools.
- Structure responses clearly with Markdown:
  - **🌾 Crop / Location Summary**
  - **🧪 Required Soil Profile (pH, N, P, K)**
  - **🌟 ML Recommendations & Yield Forecasts**
  - **💡 Agronomic Advice & Management**
- Use encouraging, practical, farmer-friendly language.
"""


async def _execute_tool(tool_name: str, args: dict) -> Any:
    """Execute the requested agent tool and return result."""

    if tool_name == "get_crop_requirements":
        crop = args.get("crop", "")
        result = ml_service.get_crop_requirements(crop)
        return result

    elif tool_name == "get_location_soil_profile":
        loc = args.get("location", "")
        state, lat, lon = ml_service.resolve_location(loc)
        soil = ml_service.get_soil_for_state(state)
        return {
            "success": True,
            "data": {
                "queried_location": loc,
                "resolved_state": state,
                "coordinates": {"latitude": lat, "longitude": lon},
                "soil_profile": soil,
            },
        }

    elif tool_name == "recommend_better_crops_for_location":
        loc = args.get("location", "")
        current_crop = args.get("current_crop")
        season = args.get("season", "Kharif")
        area = float(args.get("land_area_ha", 1.0))

        state, lat, lon = ml_service.resolve_location(loc)
        soil = ml_service.get_soil_for_state(state)

        # Get weather
        weather_data = {}
        try:
            weather_res = await get_weather_for_location(lat, lon)
            temp = weather_res.get("annual_est_temp_c") or weather_res.get("current_temperature_c", 25.0)
            hum = weather_res.get("annual_est_humidity_pct") or weather_res.get("current_humidity_pct", 65.0)
            rain = weather_res.get("annual_est_rainfall_mm") or 1000.0
            weather_data = {
                "temperature": temp,
                "humidity": hum,
                "rainfall": rain,
            }
        except Exception:
            weather_data = {"temperature": 26.0, "humidity": 70.0, "rainfall": 950.0}

        # 1. Run Crop Recommendation Classifier
        rec_input = {
            "N": soil.get("N", 75.0),
            "P": soil.get("P", 40.0),
            "K": soil.get("K", 30.0),
            "ph": soil.get("pH", 6.8),
            "temperature": weather_data["temperature"],
            "humidity": weather_data["humidity"],
            "rainfall": weather_data["rainfall"],
            "state": state,
        }
        rec_result = ml_service.recommend_crop(rec_input)

        # 2. Predict Yield for Top Recommended Crops
        forecasts = []
        top_crops = rec_result.get("top_crops", [])
        for item in top_crops[:3]:
            crop_name = item["crop"]
            try:
                y_res = ml_service.predict_yield({
                    "crop": crop_name.capitalize(),
                    "state": state,
                    "season": season,
                    "area": area,
                    "fertilizer": area * 400.0,
                    "pesticide": area * 20.0,
                    "year": 2024,
                    **soil,
                })
                forecasts.append({
                    "crop": crop_name,
                    "match_score": item["score"],
                    "predicted_yield_tonnes_ha": y_res["predicted_yield_tonnes_ha"],
                })
            except Exception:
                forecasts.append({
                    "crop": crop_name,
                    "match_score": item["score"],
                    "predicted_yield_tonnes_ha": None,
                })

        # 3. If user asked about a specific current crop, evaluate it
        current_crop_eval = None
        if current_crop:
            try:
                curr_y = ml_service.predict_yield({
                    "crop": current_crop.capitalize(),
                    "state": state,
                    "season": season,
                    "area": area,
                    "fertilizer": area * 400.0,
                    "pesticide": area * 20.0,
                    "year": 2024,
                    **soil,
                })
                current_crop_eval = {
                    "crop": current_crop,
                    "predicted_yield_tonnes_ha": curr_y["predicted_yield_tonnes_ha"],
                }
            except Exception:
                current_crop_eval = {"crop": current_crop, "predicted_yield_tonnes_ha": None}

        return {
            "success": True,
            "data": {
                "location": loc,
                "state": state,
                "soil_profile": soil,
                "weather_conditions": weather_data,
                "best_recommended_crop": rec_result.get("recommended_crop"),
                "top_recommendations": forecasts,
                "current_crop_analysis": current_crop_eval,
            },
        }

    elif tool_name == "get_weather":
        lat = args["latitude"]
        lon = args["longitude"]
        try:
            weather = await get_weather_for_location(lat, lon)
            return {"success": True, "data": weather}
        except Exception as e:
            return {"success": False, "error": str(e)}

    elif tool_name == "predict_yield":
        try:
            result = ml_service.predict_yield(args)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    elif tool_name == "recommend_crop":
        try:
            result = ml_service.recommend_crop(args)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    elif tool_name == "compare_crops":
        crops = args.get("crops", [])
        results = []
        for crop in crops:
            try:
                r = ml_service.predict_yield({**args, "crop": crop})
                results.append({"crop": crop, "predicted_yield": r["predicted_yield_tonnes_ha"]})
            except Exception as e:
                results.append({"crop": crop, "error": str(e)})
        return {"success": True, "data": {"comparisons": results}}

    elif tool_name == "get_known_options":
        return {
            "success": True,
            "data": {
                "crops":   ml_service.get_known_crops(),
                "states":  ml_service.get_known_states(),
                "seasons": ml_service.get_known_seasons(),
                "recommendable_crops": ml_service.get_recommendable_crops(),
            },
        }

    return {"success": False, "error": f"Unknown tool: {tool_name}"}


async def _try_gemini_agent(
    user_message: str,
    session_id: str,
    context: Optional[dict],
    api_key: str,
    models: list,
) -> dict:
    genai.configure(api_key=api_key)

    for model_name in models:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                tools=[{"function_declarations": AGENT_TOOLS}],
            )

            messages = []
            if context:
                messages.append({
                    "role": "user",
                    "parts": [f"[Context from previous interaction]: {json.dumps(context)}"],
                })
            messages.append({"role": "user", "parts": [user_message]})

            tool_calls_made = []
            structured_data = {}

            # Agent loop
            for _ in range(5):
                response = await asyncio.to_thread(model.generate_content, messages)
                if not response or not getattr(response, "candidates", None):
                    break

                candidate = response.candidates[0]

                function_calls = [
                    part.function_call
                    for part in candidate.content.parts
                    if hasattr(part, "function_call") and part.function_call
                ]

                if not function_calls:
                    final_text = "".join(
                        part.text for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    )
                    if final_text and final_text.strip():
                        return {
                            "reply": final_text,
                            "session_id": session_id,
                            "tool_calls_made": tool_calls_made,
                            "structured_data": structured_data if structured_data else None,
                        }
                    break

                tool_results = []
                for fc in function_calls:
                    tool_name = fc.name
                    args = dict(fc.args)
                    logger.info(f"Agent ({model_name}) calling tool: {tool_name}({args})")
                    tool_calls_made.append(tool_name)

                    result = await _execute_tool(tool_name, args)
                    if result.get("success") and result.get("data"):
                        structured_data[tool_name] = result["data"]

                    tool_results.append({
                        "function_response": {
                            "name": tool_name,
                            "response": result,
                        }
                    })

                messages.append({"role": "model", "parts": candidate.content.parts})
                messages.append({"role": "user", "parts": tool_results})

            return {
                "reply": "Processed your request using agricultural decision models.",
                "session_id": session_id,
                "tool_calls_made": tool_calls_made,
                "structured_data": structured_data if structured_data else None,
            }

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Model {model_name} failed: {err_str[:120]}")
            # If 429 quota error or invalid key, immediately fail fast and trigger local ML fallback
            if any(k in err_str.lower() for k in ["429", "quota", "resourceexhausted", "invalid", "not found", "404"]):
                raise e

    raise RuntimeError("Gemini models unavailable.")


async def agent_chat(
    user_message: str,
    session_id: Optional[str] = None,
    context: Optional[dict] = None,
) -> dict:
    """
    Main agent entry point. Sends user message to Gemini with function calling enabled.
    Falls back instantly to local ML & agronomic pipelines if quota is exceeded or timeout occurs.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here" or len(api_key) < 10:
        logger.info("GEMINI_API_KEY missing or placeholder. Using local ML & agronomic decision fallback.")
        return await _local_agronomic_fallback(user_message, session_id)

    raw_models = [
        os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ]
    seen = set()
    models = [m for m in raw_models if not (m in seen or seen.add(m))]

    try:
        # 6-second strict max timeout for Gemini API calls to ensure front-end never times out
        return await asyncio.wait_for(
            _try_gemini_agent(user_message, session_id, context, api_key, models),
            timeout=6.0,
        )
    except Exception as e:
        logger.warning(f"Gemini API execution bypassed ({type(e).__name__}: {str(e)[:100]}). Falling back to local ML engine.")
        return await _local_agronomic_fallback(user_message, session_id)



async def _local_agronomic_fallback(user_message: str, session_id: str) -> dict:
    """Intelligent offline fallback executing local ML pipelines directly."""
    msg_lower = user_message.lower()

    # Known Indian locations (states & major cities)
    known_locations = [
        "punjab", "haryana", "maharashtra", "tamil nadu", "uttar pradesh", "karnataka",
        "andhra pradesh", "gujarat", "madhya pradesh", "west bengal", "bihar", "rajasthan",
        "kerala", "telangana", "odisha", "assam", "jharkhand", "chhattisgarh",
        "ludhiana", "amritsar", "jalandhar", "pune", "nagpur", "chennai", "coimbatore",
        "lucknow", "kanpur", "bengaluru", "vijayawada", "ahmedabad", "surat", "bhopal",
        "kolkata", "patna", "jaipur", "kochi", "hyderabad", "bhubaneswar", "guwahati", "ranchi"
    ]

    # Standard crop aliases mapping
    crop_aliases = {
        "cotton": "Cotton",
        "rice": "Rice",
        "wheat": "Wheat",
        "maize": "Maize",
        "chickpea": "Chickpea",
        "banana": "Banana",
        "mango": "Mango",
        "grapes": "Grapes",
        "watermelon": "Watermelon",
        "muskmelon": "Muskmelon",
        "apple": "Apple",
        "orange": "Orange",
        "papaya": "Papaya",
        "coconut": "Coconut",
        "coffee": "Coffee",
        "jute": "Jute",
        "blackgram": "Blackgram",
        "lentil": "Lentil",
        "mungbean": "Mungbean",
        "mothbeans": "Mothbeans",
        "pigeonpeas": "Pigeonpeas",
        "pomegranate": "Pomegranate",
        "kidneybeans": "Kidneybeans",
    }

    detected_crop = None
    for alias, clean_name in crop_aliases.items():
        if alias in msg_lower:
            detected_crop = clean_name
            break

    if not detected_crop:
        known_crops = ml_service.get_known_crops()
        for crop in known_crops:
            clean = crop.split("(")[0].strip()
            if clean.lower() in msg_lower:
                detected_crop = clean
                break

    detected_location = None
    for loc in known_locations:
        if loc in msg_lower:
            detected_location = loc
            break

    # Case A: Location is present OR user explicitly asks for recommendation / location advice
    is_rec_query = any(w in msg_lower for w in ["recommend", "better crop", "suitable", "best crop", "forecast", "yield", "location", "state", "grow"])

    if detected_location or (is_rec_query and not (("ph" in msg_lower or "npk" in msg_lower or "nutrient" in msg_lower or "requirement" in msg_lower) and detected_crop and not detected_location)):
        loc_target = detected_location or user_message
        loc_res = await _execute_tool("recommend_better_crops_for_location", {
            "location": loc_target,
            "current_crop": detected_crop,
        })

        if loc_res.get("success"):
            d = loc_res["data"]
            soil = d["soil_profile"]
            top = d["top_recommendations"]
            best_crop = d["best_recommended_crop"]

            top_list = "\n".join([
                f"* **{item['crop'].capitalize()}**: Match Score `{round(item['match_score'] * 100, 1)}%`"
                + (f" | Forecast Yield: **{item['predicted_yield_tonnes_ha']} t/ha**" if item['predicted_yield_tonnes_ha'] else "")
                for item in top
            ])

            comparison_block = ""
            if d.get("current_crop_analysis") and d["current_crop_analysis"].get("crop"):
                curr = d["current_crop_analysis"]
                curr_name = curr["crop"].capitalize()
                curr_yield = curr.get("predicted_yield_tonnes_ha")
                yield_str = f"**{curr_yield} t/ha**" if curr_yield else "N/A"
                comparison_block = (
                    f"### ⚖️ Head-to-Head Comparison for **{curr_name}** in {d['state']}\n"
                    f"* Your Intended Crop (**{curr_name}**): Estimated Yield = {yield_str}\n"
                    f"* Top ML Recommended Alternative (**{best_crop.capitalize()}**): Match Score = `{round(top[0]['match_score']*100, 1)}%` | Yield = **{top[0]['predicted_yield_tonnes_ha']} t/ha**\n\n"
                )

            reply = (
                f"### 📍 Agricultural Decision & Crop Recommendation for **{d['state']}**\n\n"
                f"{comparison_block}"
                f"**🧪 Regional Soil Profile:**\n"
                f"* Nitrogen (N): **{soil.get('N')} kg/ha** | Phosphorus (P): **{soil.get('P')} kg/ha** | "
                f"Potassium (K): **{soil.get('K')} kg/ha** | Soil pH: **{soil.get('pH')}**\n\n"
                f"**🌟 Machine Learning Recommendations (Ranked):**\n"
                f"{top_list}\n\n"
                f"**💡 Agronomic Summary & Advice:**\n"
                f"Based on the empirical soil chemistry (pH {soil.get('pH')}) and regional climate of {d['state']}, **{best_crop.capitalize()}** is "
                f"the most optimal crop to cultivate for maximum economic yield and optimal nutrient efficiency."
            )
            return {
                "reply": reply,
                "session_id": session_id,
                "tool_calls_made": ["recommend_better_crops_for_location"],
                "structured_data": d,
            }

    # Case B: Crop-specific soil requirements / NPK / pH query
    if detected_crop or "ph" in msg_lower or "npk" in msg_lower or "nutrient" in msg_lower or "soil" in msg_lower:
        target_crop = detected_crop or "rice"
        req = ml_service.get_crop_requirements(target_crop)
        if req.get("success"):
            d = req["data"]
            ph = d["soil_pH"]
            n = d["nitrogen_N_kg_ha"]
            p = d["phosphorus_P_kg_ha"]
            k = d["potassium_K_kg_ha"]
            adv = d.get("agronomic_advice", "")
            reply = (
                f"### 🌾 Soil & Nutrient Requirements for **{d['crop']}**\n\n"
                f"Here is the scientifically verified soil profile and climatic parameters:\n\n"
                f"**🧪 Soil Chemistry (pH & N-P-K)**\n"
                f"* **Soil pH:** **{ph['optimal']}** (tolerable range: `{ph['range']}`)\n"
                f"* **Nitrogen (N):** **{n['optimal']} kg/ha** (range: `{n['range']}`)\n"
                f"* **Phosphorus (P):** **{p['optimal']} kg/ha** (range: `{p['range']}`)\n"
                f"* **Potassium (K):** **{k['optimal']} kg/ha** (range: `{k['range']}`)\n\n"
                f"**🌤️ Climatic Needs**\n"
                f"* **Temperature:** {d['climate']['temperature_c']}\n"
                f"* **Humidity:** {d['climate']['humidity_pct']}\n"
                f"* **Rainfall:** {d['climate']['rainfall_mm']}\n\n"
                f"**💡 Agronomic Management & Advice**\n"
                f"{adv}"
            )
            return {
                "reply": reply,
                "session_id": session_id,
                "tool_calls_made": ["get_crop_requirements"],
                "structured_data": {"crop_requirements": d},
            }

    # Case C: General fallback
    return {
        "reply": (
            "### 🤖 Agentic AI Agricultural Decision Assistant\n\n"
            "I can assist you with:\n"
            "1. **🌾 Crop Soil & Nutrient Requirements**: Ask for ideal pH, N, P, K for any crop (e.g. Cotton, Rice, Wheat, Maize).\n"
            "2. **📍 Location-Based Crop Recommendations**: Ask for optimal crops in any state (e.g. Punjab, Maharashtra, Tamil Nadu).\n"
            "3. **⚖️ Crop Suitability Comparisons**: Ask whether a crop is suitable for your state compared to alternatives.\n"
            "4. **📊 Yield Forecasts**: Predict expected tonnes/hectare based on regional soil and weather data."
        ),
        "session_id": session_id,
        "tool_calls_made": [],
        "structured_data": None,
    }


