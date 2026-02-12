# REST API — Quick Reference Guide ✅

A concise, interview-ready guide to REST APIs: principles, HTTP methods, status codes, URL design, and best practices.

## Table of contents
- [What is a REST API?](#-what-is-a-rest-api-)
- [Key Principles of REST](#-key-principles-of-rest-)
- [HTTP Methods (CRUD)](#-http-methods-crud-)
- [HTTP Status Codes — Essentials](#-http-status-codes--essentials-)
- [REST URL Design](#-rest-url-design-)
- [Request & Response](#-request--response-)
- [Query vs Path vs Body](#-query-vs-path-vs-body-)
- [Statelessness — Core Concept](#-statelessness--core-concept-)
- [Examples](#examples-)
- [Interview Tips & One-liners](#-interview-tips--one-liners-)

---

## 1️⃣ What is a REST API? 💡

**REST** = REpresentational State Transfer — an architectural style for web APIs.

- Stateless: each request contains all information needed
- Resource-based: resources are identified by URLs (e.g., `/users/1`)
- Usually uses JSON for data exchange

Example:

```http
GET /users         # get all users
GET /users/123     # get user with ID 123
POST /users        # create a new user (body contains JSON)
```

> Interview line: “A REST API is a web service following REST principles: stateless, resource-oriented, HTTP-based, typically using JSON.”

---

## 2️⃣ Key Principles of REST 🔑

- **Stateless** — Server does not store client session/state.
- **Client–Server** — Separation of concerns.
- **Uniform Interface** — Consistent URLs, methods, response formats.
- **Cacheable** — Responses can be cached when appropriate.
- **Layered System** — Requests may pass through intermediaries.
- **Code on Demand** (optional) — Server may send executable code (rare).

**Quick one-liner:** “REST principles (stateless, client-server, uniform interface, cacheable, layered) enable scalable, maintainable APIs.”

---

## 3️⃣ HTTP Methods (CRUD) 🛠️

| Method | CRUD     | Use case
|--------|----------|-------------------------------
| GET    | Read     | Fetch resources (safe, idempotent)
| POST   | Create   | Create resources (not idempotent)
| PUT    | Replace  | Replace an entire resource (idempotent)
| PATCH  | Modify   | Partial update (not always idempotent)
| DELETE | Delete   | Remove resource (idempotent)

Example:

```http
POST /users                     # create user -> 201 Created
GET /users/1                    # get user -> 200 OK
PATCH /users/1                  # update some fields -> 200 OK
DELETE /users/1                 # delete user -> 204 No Content
```

---

## 4️⃣ HTTP Status Codes — Essentials ✅

| Code | Meaning
|------|-------------------------------
| 200  | OK (success)
| 201  | Created (resource created)
| 204  | No Content (successful, no body)
| 400  | Bad Request (client error)
| 401  | Unauthorized (missing/invalid auth)
| 403  | Forbidden (authenticated, not allowed)
| 404  | Not Found (resource missing)
| 422  | Unprocessable Entity (validation error)
| 500  | Internal Server Error (server side)

**Tip:** Use correct status codes — don't always return 200 for errors.

---

## 5️⃣ REST URL Design 🧭

- Use **nouns** (resources), not verbs. Prefer plural: `/users` not `/user`.
- Use path params for resource identity: `/users/{id}`.
- Use query params for filtering/pagination: `/users?limit=10&skip=5`.
- Version your API: e.g., `/api/v1/users`.

Bad: `/getUsers`, `/createUser`
Good: `GET /users`, `POST /users`

---

## 6️⃣ Request & Response ✉️

Request components:
- Method, URL, Headers (Content-Type, Authorization), Optional body (JSON)

Response components:
- Status code, Headers, Body (usually JSON)

Example request/response:

```http
POST /users
Content-Type: application/json

{ "name": "Alice", "email": "alice@example.com" }

# Response
201 Created
{ "id": 1, "name": "Alice" }
```

---

## 7️⃣ Query vs Path vs Body 🔀

- **Path** `/users/{id}` — identifies a specific resource.
- **Query** `/users?limit=10` — filtering, sorting, pagination.
- **Body** — send resource data for POST/PUT/PATCH.

One-liner: “Path identifies, query modifies the request, body carries data.”

---

## 8️⃣ Statelessness — Core Concept 🔐

Each request is independent. Authentication (JWT/API key) must be sent with every request, e.g.: `Authorization: Bearer <token>`.

Why: enables horizontal scaling and simpler architecture.

---

## 9️⃣ Why REST is Popular ⭐

- Simple (HTTP + JSON)
- Language-agnostic
- Works well for web, mobile, microservices
- Supports caching and scalable architectures

**Interview answer:** “REST is stateless, resource-oriented, uses standard HTTP methods and status codes, and typically exchanges JSON. It is scalable and language-agnostic.”

## Examples 🧪

### Curl examples

GET all users

```bash
curl -i http://localhost:8000/users
```

Expected (200 OK):

```http
HTTP/1.1 200 OK
Content-Type: application/json

[ { "id": 1, "name": "Alice" } ]
```

Create user (POST):

```bash
curl -i -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Bob","email":"bob@example.com"}'
```

Expected (201 Created):

```http
HTTP/1.1 201 Created
Content-Type: application/json

{ "id": 2, "name": "Bob", "email": "bob@example.com" }
```

Partial update (PATCH):

```bash
curl -i -X PATCH http://localhost:8000/users/2 \
  -H "Content-Type: application/json" \
  -d '{"email":"bob@new.com"}'
```

Expected (200 OK): updated user object

Delete (DELETE):

```bash
curl -i -X DELETE http://localhost:8000/users/2
```

Expected (204 No Content)

### Authorization example

Add an Authorization header with a JWT:

```http
Authorization: Bearer <your-jwt-token>
```

### FastAPI sample (validation & status codes)

```python
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, EmailStr

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: EmailStr

users = []

@app.post('/users', status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate):
    new = user.dict()
    new['id'] = len(users) + 1
    users.append(new)
    return new

@app.get('/users')
def list_users():
    return users

@app.patch('/users/{user_id}')
def update_user(user_id: int, user: dict):
    for u in users:
        if u['id'] == user_id:
            u.update(user)
            return u
    raise HTTPException(status_code=404, detail='User not found')

# FastAPI automatically returns 422 Unprocessable Entity for invalid input
```

### Quick cheat-sheet

- Use correct status codes (201 for created, 204 for delete)
- Path = resource id, Query = filters, Body = data
- Send `Authorization` header with token for auth

---

## Interview Tips & One-liners 🧠

- Statelessness: “Server does not store client context; each request contains all required info.”
- Uniform interface: “Use standard HTTP methods and predictable resource URLs.”
- Status codes matter: “Use 2xx for success, 4xx for client errors, 5xx for server errors.”

---

© Generated reference — keep this file in your repo for quick interview prep.
