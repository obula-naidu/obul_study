REST (Representational State of Resource) is an architectural style for designing networked applications. It's a way to communicate between systems and services over the web, using standard HTTP methods (GET, POST, PUT, DELETE, etc.) to manipulate resources.

The core principles of REST are:

1. **Resources**: Everything in REST can be identified by a unique identifier, such as a URL. These resources can be data, objects, or even actions.
2. **Client-Server Architecture**: The client and server are separate entities. The client makes requests to the server to perform operations on resources, while the server handles those requests and returns responses.      
3. **Stateless**: Each request from the client to the server must contain all the information necessary to complete that action. The server does not maintain any information about the state of the request or the client.     
4. **Cacheable**: Responses from the server can be cached by the client, reducing the need for repeated requests.
5. **Uniform Interface**: A uniform interface is used to communicate between systems and services. This includes standard HTTP methods, URI schemes, and data formats.

Some key concepts in REST include:

* **Resource-based**: Everything is a resource, and resources are manipulated using HTTP methods (e.g., GET, POST, PUT, DELETE).
* **Request-Response Model**: The client makes a request to the server, which responds with a result.
* **HTTP Methods**:
        + GET: Retrieve data
        + POST: Create new data
        + PUT: Update existing data
        + DELETE: Delete data

REST is designed to be:

* Platform-independent (can work across different operating systems and devices)
* Language-agnostic (can be implemented in various programming languages)
* Scalable (can handle a large number of requests and responses)

Overall, REST provides a flexible, scalable, and maintainable way to design networked applications, making it a widely used architectural style for web services.