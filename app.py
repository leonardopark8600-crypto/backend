from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config_settings import get_cors_config
import websocket_routes
import sbml_routes
import opt_addon.optimize_routes as optimize_routes

# Crear app
app = FastAPI(title="SBML Simulator API")

# Configurar CORS
cors_config = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)

# Registrar routers
app.include_router(websocket_routes.router)
app.include_router(sbml_routes.router)
app.include_router(optimize_routes.router, prefix="/opt")  # 👈 rutas de optimización

# Endpoint base
@app.get("/")
def home():
    return {"message": "Servidor de contador en línea activo 🚀"}

