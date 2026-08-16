# -*- coding: utf-8 -*-
"""
src/utils/gemini_api_tester.py

Utility to test and verify a Gemini API Key against Google's Gemini API endpoints.
Supports both modern `google.genai` SDK and legacy `google.generativeai` fallback.
"""

from __future__ import annotations

import os
from typing import Tuple, List, Optional
from src.core.result import Result


def verify_gemini_api_key(api_key: str) -> Result[Tuple[str, List[str]], str]:
    """
    Test if a Gemini API Key is valid and functional.

    Returns:
        Result.Ok((working_model_name, list_of_available_models)) if successful.
        Result.Err(error_message) if verification fails.
    """
    key = (api_key or "").strip()
    if not key:
        return Result.Err("API Key không được để trống!")

    # Check ShopAIKey REST API endpoint first if key starts with sk-
    if key.startswith("sk-"):
        try:
            import requests
            url = f"https://api.shopaikey.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": "Xin chào! Trả lời 'OK' để xác nhận."}]}]
            }
            res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
            if res.status_code == 200:
                return Result.Ok(("gemini-2.5-flash (ShopAIKey)", ["gemini-2.5-flash"]))
            else:
                err_text = res.json().get("error", {}).get("message", res.text)
                return Result.Err(f"ShopAIKey Error ({res.status_code}): {err_text}")
        except Exception as exc:
            return Result.Err(f"Không thể kết nối tới ShopAIKey: {exc}")

    # Prefer google.genai SDK
    try:
        import google.genai as genai
        client = genai.Client(api_key=key)
        
        # Test candidate models in order
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]
        
        working_model = None
        last_error = None
        
        for m in candidate_models:
            try:
                res = client.models.generate_content(
                    model=m,
                    contents="Xin chào! Trả lời 'OK' để xác nhận API Key hoạt động.",
                )
                if res and res.text:
                    working_model = m
                    break
            except Exception as exc:
                last_error = str(exc)
                continue

        if working_model:
            return Result.Ok((working_model, [working_model]))
        
        # If specific candidate models failed, try listing models from API to check available ones
        try:
            available_models = []
            for m in client.models.list():
                if hasattr(m, "name") and m.name:
                    available_models.append(m.name)
            if available_models:
                for raw_m in available_models[:10]:
                    clean_m = raw_m.replace("models/", "")
                    try:
                        res = client.models.generate_content(
                            model=clean_m,
                            contents="Xin chào!",
                        )
                        if res and res.text:
                            return Result.Ok((clean_m, available_models))
                    except Exception as sub_exc:
                        last_error = str(sub_exc)
        except Exception:
            pass

        err_str = str(last_error or "")
        if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in err_str or "UNAUTHENTICATED" in err_str:
            return Result.Err(
                f"Lỗi xác thực API Key (401 UNAUTHENTICATED).\n\n"
                f"Nguyên nhân: Key dạng 'AQ.Ab8...' trên tài khoản này yêu cầu kích hoạt 'Generative Language API' "
                f"trong Google Cloud Console hoặc chọn 'Create API Key in new project' trên aistudio.google.com.\n\n"
                f"Chi tiết từ Google: {err_str}"
            )
        elif "PERMISSION_DENIED" in err_str or "403" in err_str:
            return Result.Err(
                f"Lỗi quyền truy cập API Key (403 PERMISSION_DENIED).\n\n"
                f"Dự án Google Cloud chưa được bật quyền Generative Language API.\n\n"
                f"Chi tiết từ Google: {err_str}"
            )
        elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            return Result.Err(
                f"Key đã vượt quá hạn ngạch (429 RESOURCE_EXHAUSTED / Quota Exceeded).\n\n"
                f"Vui lòng đợi vài phút hoặc tạo Key mới trên aistudio.google.com.\n\n"
                f"Chi tiết từ Google: {err_str}"
            )

        return Result.Err(f"API Key bị từ chối hoặc chưa có quyền sử dụng model.\nChi tiết lỗi: {last_error}")

    except ImportError:
        pass

    # Fallback to google.generativeai if google.genai is not available
    try:
        import google.generativeai as g_old
        g_old.configure(api_key=key)
        
        models_list = []
        try:
            for m in g_old.list_models():
                if 'generateContent' in getattr(m, 'supported_generation_methods', []):
                    models_list.append(m.name.replace("models/", ""))
        except Exception as exc:
            return Result.Err(f"Không thể kết nối đến Gemini API (Lỗi API Key hoặc Mạng):\n{exc}")

        for m_name in models_list:
            try:
                mod = g_old.GenerativeModel(m_name)
                resp = mod.generate_content("Xin chào!")
                if resp and resp.text:
                    return Result.Ok((m_name, models_list))
            except Exception as exc:
                last_error = str(exc)
                continue

        return Result.Err(f"API Key không thể gọi model nào thành công. Chi tiết: {last_error}")

    except Exception as exc:
        return Result.Err(f"Lỗi khởi tạo Gemini SDK: {exc}")
