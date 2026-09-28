DIRETORIO_SAIDA = "saida"

#WHISPER
DISPOSITIVO_WHISPER = "auto"
MODELO_WHISPER = "base"
FILTRO_WHISPER_VAD = False
PARAMETROS_WHISPER_VAD = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }